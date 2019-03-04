import pandas as pd 
import datetime
import time
import os

import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, auc, accuracy_score, roc_curve, balanced_accuracy_score

from plot_utils import plot_confusion_matrix, plot_model_family_learning_curves, plot_opt_prob_curves
from etl_utils import *

import mlrose

np.seterr(over='ignore') #ignore overflow in exponent warnings

np.random.seed(27)
#general params - plotting and stdout
PLOT_ACTION = None # (None, 'save', 'show') - default to None to avoid issues with matplotlib depending on OS TODO: set to None
PLOT_BATCH_LC = False #this will be overwritten if PLOT_ACTION is "None"
ITERATION_OUTPUT = False # True or False, to print updates as each algorithm is iterating TODO: set to False
OUTPUT_CSV = False # true or false to write CSV with summaries. If True, must have write access to local directory

# randomized optimization for neural network params
N_RESTARTS = 3 # number of random restarts to do with randomized hill climbing.
IMPROVEMENT_TOLERANCE = 0.00001 #this helps determine if convergence has happened
N_MAX_ITER = 500
N_MAX_ATTEMPTS = 50

# randomized optimization for discrete optimization problems
DO_MAX_ITER = 10000
DO_MAX_ATTEMPTS = 20 

#dataset handling and cleaning
USE_DATASET = 'aps' #one of ('spam', 'aps')
BALANCE_METHOD = 'downsample' # (int, 'downsample' or 'upsample') #downsample to balance classes
N_LC_CHUNKS = 3 #number of chunks for learning curve data segmentation, if PLOT_BATCH_LC = True
N_CV = 0.2 # percentage to use as validation in learning curve, if PLOT_BATCH_LC = True

# FULL DISCLOSURE, THESE MONKEY-PATCHED METHODS ARE DIRECTLY STOLEN FROM mlrose: https://github.com/gkhayes/mlrose
# (a dependency of this project), with some modifications to algorithms and neural network fitting, along with tracking information for each algorithm. 
# The following 5 methods are monkey-patched from the original source code 
# 1. NeuralNetwork.fit
# 2. algorithms.random_hill_climb, 
# 3. algorithms.simulated_annealing
# 4. algorithms.genetic_alg
# 5. algorithms.mimic
# A new method is patched in as well, random_sa_neighbor, which allows for setting a variable step size based on temperature for continuous problems.

def fit(self, X, y, init_weights=None):
    # Make sure y is an array and not a list
    y = np.array(y)

    # Convert y to 2D if necessary
    if len(np.shape(y)) == 1:
        y = np.reshape(y, [len(y), 1])

    # Verify X and y are the same length
    if not np.shape(X)[0] == np.shape(y)[0]:
        raise Exception('The length of X and y must be equal.')

    # Determine number of nodes in each layer
    input_nodes = np.shape(X)[1] + self.bias
    output_nodes = np.shape(y)[1]
    node_list = [input_nodes] + self.hidden_nodes + [output_nodes]

    num_nodes = 0

    for i in range(len(node_list) - 1):
        num_nodes += node_list[i]*node_list[i+1]

    if init_weights is not None and len(init_weights) != num_nodes:
        raise Exception("""init_weights must be None or have length %d"""
                        % (num_nodes,))

    # Initialize optimization problem
    fitness = mlrose.neural.NetworkWeights(X, y, node_list, self.activation, self.bias,
                                self.is_classifier, learning_rate=self.lr)

    problem = mlrose.opt_probs.ContinuousOpt(num_nodes, fitness, maximize=False,
                            min_val=-1*self.clip_max,
                            max_val=self.clip_max, step=self.lr)

    if self.algorithm == 'random_hill_climb':
        if init_weights is None:
            #this sets the init weights permanently for each restart
            init_weights = np.random.uniform(-1, 1, num_nodes)
        fitted_weights, loss, loss_list, func_eval_list = random_hill_climb(
                                            problem,
                                            max_attempts=self.max_attempts, max_iters=self.max_iters,
                                            restarts=N_RESTARTS, init_state=init_weights)

    elif self.algorithm == 'simulated_annealing':
        if init_weights is None:
            init_weights = np.random.uniform(-1, 1, num_nodes)
        fitted_weights, loss, loss_list, func_eval_list = simulated_annealing(
                                            problem,
                                            schedule=self.schedule, max_attempts=self.max_attempts,
                                            max_iters=self.max_iters, init_state=init_weights)

    elif self.algorithm == 'genetic_alg':
        fitted_weights, loss, loss_list, func_eval_list = genetic_alg(
                                            problem,
                                            pop_size=self.pop_size, mutation_prob=self.mutation_prob,
                                            max_attempts=self.max_attempts, max_iters=self.max_iters)

    else:  # Gradient descent case
        if init_weights is None:
            init_weights = np.random.uniform(-1, 1, num_nodes)
        fitted_weights, loss = gradient_descent(
            problem,
            max_attempts=self.max_attempts, max_iters=self.max_iters,
            init_state=init_weights)

    # Save fitted weights and node list
    self.node_list = node_list
    self.fitted_weights = fitted_weights
    self.loss = loss
    self.loss_list = loss_list
    self.func_eval_list = func_eval_list
    self.output_activation = fitness.get_output_activation()

def random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=10,
                      init_state=None):
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    best_fitness = -1*np.inf
    best_state = None

    fitness_list = []
    func_eval_count = 0
    print('\nworking on: RHC')
    
    for i in range(restarts + 1):
        
        # Initialize optimization problem and attempts counter
        problem.reset()
        # else:
        #     print('initial state is already set at:', init_state)
        #     problem.set_state(init_state)

        attempts = 0
        iters = 0
        no_improve_iters = 0
        while (attempts < max_attempts) and (iters < max_iters) and (no_improve_iters < max_attempts):
            iters += 1
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)
            func_eval_count += 1
            if ITERATION_OUTPUT:
                print('iter: {} - next fitness: {:0.3e}'.format(iters,next_fitness), end='\r', flush=True)

            # If best neighbor is an improvement,
            # move to that state and reset attempts counter
            if next_fitness >= problem.get_fitness():
                #slight adjustment, allows transfer to next state, even if it is the exact same as current state
                if abs(next_fitness-problem.get_fitness()) < IMPROVEMENT_TOLERANCE:
                    # no improvement iters
                    no_improve_iters += 1
                else:
                    no_improve_iters = 0
                problem.set_state(next_state)

                attempts = 0
            else:
                attempts += 1

            #this should always be the same as the last move, unless this fitness eval was better than the last
            fitness_list.append(problem.get_fitness())
        

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()
    print('')
    best_fitness = problem.get_maximize()*best_fitness
    func_eval_list = fitness_list #func evals is the same as the fitness evaluation list
    return best_state, best_fitness, fitness_list, func_eval_count

def simulated_annealing(problem, schedule, max_attempts=10,
                        max_iters=np.inf, init_state=None):
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    attempts = 0
    iters = 0
    no_improve_iters = 0
    fitness_list = []
    func_eval_count = 0
    print('\nworking on: SA')
    while (attempts < max_attempts) and (iters < max_iters) and (no_improve_iters < max_attempts):
        temp = schedule.evaluate(iters)
        iters += 1

        if temp <= 0:
            print('temperature went negative, stopping now')
            break

        #if it's a discrete variable, use original random_neighbor method
        if isinstance(problem, mlrose.opt_probs.DiscreteOpt) or issubclass(type(problem), mlrose.opt_probs.DiscreteOpt):
            next_state = problem.random_neighbor()
        else:
            # otherwise
            # Find random neighbor and evaluate fitness - THIS SHOULD BE GLOBAL RANDOM NEIGHBOR
            next_state = problem.random_sa_neighbor(current_temp=temp)

        next_fitness = problem.eval_fitness(next_state)
        func_eval_count += 1

        # Calculate delta E and change prob
        current_fitness = problem.get_fitness()
        delta_e = next_fitness - current_fitness
    
        prob = np.exp(delta_e/temp)
        if ITERATION_OUTPUT:
            print('iter: {} - next fitness: {:0.3e}'.format(iters,next_fitness), end='\r', flush=True)

        # If best neighbor is an improvement or random value is less
        # than prob, move to that state and reset attempts counter 
        rando = np.random.uniform()
        if (delta_e > 0) or (rando < prob):
            if abs(delta_e) < IMPROVEMENT_TOLERANCE:
                # no improvement iters
                no_improve_iters += 1
            else:
                no_improve_iters = 0
            problem.set_state(next_state)
            attempts = 0
        else:
            attempts += 1

        fitness_list.append(problem.get_fitness())

    print('')

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    return best_state, best_fitness, fitness_list, func_eval_count

def genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10,
                max_iters=np.inf):
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    fitness_list = []
    func_eval_count = 0

    print('\nworking on: GA')
    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []

        for _ in range(pop_size):
            # Select parents
            selected = np.random.choice(pop_size, size=2,
                                        p=problem.get_mate_probs())
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        next_gen = np.array(next_gen)

        problem.set_population(next_gen)
        func_eval_count += (len(next_gen)) #each member of population is evaluated

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)
        func_eval_count += 1

        if ITERATION_OUTPUT:
            print('iter: {} - next fitness: {:0.3e}'.format(iters,next_fitness), end='\r', flush=True)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0
        else:
            attempts += 1
        
        fitness_list.append(problem.get_fitness())

    print('')
    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    return best_state, best_fitness, fitness_list, func_eval_count

def mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10,
          max_iters=np.inf):
    if problem.get_prob_type() == 'continuous':
        raise Exception("""problem type must be discrete or tsp.""")

    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (keep_pct < 0) or (keep_pct > 1):
        raise Exception("""keep_pct must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0
    fitness_list = []
    func_eval_count = 0
    print('\nworking on: MIMIC')
    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Get top n percent of population
        problem.find_top_pct(keep_pct)

        # Update probability estimates
        problem.eval_node_probs()

        # Generate new sample
        new_sample = problem.sample_pop(pop_size)
        problem.set_population(new_sample)
        func_eval_count += len(new_sample)

        next_state = problem.best_child()

        next_fitness = problem.eval_fitness(next_state)
        fitness_list.append(next_fitness)
        func_eval_count += 1
        if ITERATION_OUTPUT:
            print('iter: {} - next fitness: {:0.3e}'.format(iters,next_fitness), end='\r', flush=True)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1
    print('')
    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state().astype(int)
    
    return best_state, best_fitness, fitness_list, func_eval_count

#new random neighbor function for simulated annealing, based on current temperature
def random_sa_neighbor(self, current_temp):
    while True:
        neighbor = np.copy(self.state)
        random_distro = np.random.uniform(-np.sqrt(current_temp), np.sqrt(current_temp),len(neighbor))
        neighbor += random_distro
        #clip the neighbors to max and min
        neighbor = np.clip(neighbor, self.min_val, self.max_val)
        if not np.array_equal(np.array(neighbor), self.state):
            break
    return neighbor

mlrose.opt_probs.ContinuousOpt.random_sa_neighbor = random_sa_neighbor   
mlrose.neural.NeuralNetwork.fit = fit
mlrose.algorithms.random_hill_climb = random_hill_climb
mlrose.algorithms.simulated_annealing = simulated_annealing
mlrose.algorithms.genetic_alg = genetic_alg
mlrose.algorithms.mimic = mimic

#Model generator functions that return lists of models, with different params.
def generate_rhc_models(algo_ids=None):
    rhc_nn_A = mlrose.NeuralNetwork(hidden_nodes = [100,20], activation = 'relu', \
                                algorithm = 'random_hill_climb', max_iters=N_MAX_ITER, \
                                bias = True, is_classifier = True, learning_rate = 0.1, \
                                early_stopping = True, clip_max = 1, max_attempts = N_MAX_ATTEMPTS)
    rhc_nn_A = MachineLearningModel(rhc_nn_A, 'RHC-NN', '0.1LR', 'mlrose', id=algo_ids)

    rhc_nn_B = mlrose.NeuralNetwork(hidden_nodes = [100,20], activation = 'relu', \
                                algorithm = 'random_hill_climb', max_iters=N_MAX_ITER, \
                                bias = True, is_classifier = True, learning_rate = 0.3, \
                                early_stopping = True, clip_max = 1, max_attempts = N_MAX_ATTEMPTS)
    rhc_nn_B = MachineLearningModel(rhc_nn_B, 'RHC-NN', '0.3LR', 'mlrose', id=algo_ids)
    
    rhc_nn_C = mlrose.NeuralNetwork(hidden_nodes = [100,20], activation = 'relu', \
                                algorithm = 'random_hill_climb', max_iters=N_MAX_ITER, \
                                bias = True, is_classifier = True, learning_rate = 0.5, \
                                early_stopping = True, clip_max = 1, max_attempts = N_MAX_ATTEMPTS)
    rhc_nn_C = MachineLearningModel(rhc_nn_C, 'RHC-NN', '0.5LR', 'mlrose', id=algo_ids)

    return [rhc_nn_A, rhc_nn_B, rhc_nn_C]

def generate_sa_models(algo_ids=None):
    sa_nn_A = mlrose.NeuralNetwork(hidden_nodes = [100,20], activation = 'relu', \
                                algorithm = 'simulated_annealing', max_iters=N_MAX_ITER,\
                                bias = True, is_classifier = True, \
                                early_stopping = True, clip_max = 3, max_attempts = N_MAX_ATTEMPTS,\
                                schedule=mlrose.decay.GeomDecay(init_temp=2.0, decay=0.99, min_temp=0.0))
    sa_nn_A = MachineLearningModel(sa_nn_A, 'SA-NN', '2-Temp-0.99-Decay', 'mlrose', id=algo_ids)

    sa_nn_B = mlrose.NeuralNetwork(hidden_nodes = [100,20], activation = 'relu', \
                                algorithm = 'simulated_annealing', max_iters=N_MAX_ITER,\
                                bias = True, is_classifier = True, \
                                early_stopping = True, clip_max = 3, max_attempts = N_MAX_ATTEMPTS,\
                                schedule=mlrose.decay.GeomDecay(init_temp=2.0, decay=0.999, min_temp=0.0))
    sa_nn_B = MachineLearningModel(sa_nn_B, 'SA-NN', '2-Temp-0.999-Decay', 'mlrose', id=algo_ids)  

    sa_nn_C = mlrose.NeuralNetwork(hidden_nodes = [100,20], activation = 'relu', \
                                algorithm = 'simulated_annealing', max_iters=N_MAX_ITER, \
                                bias = True, is_classifier = True, \
                                early_stopping = True, clip_max = 3, max_attempts = N_MAX_ATTEMPTS,\
                                schedule=mlrose.decay.GeomDecay(init_temp=10.0, decay=0.99, min_temp=0.0))
    sa_nn_C = MachineLearningModel(sa_nn_C, 'SA-NN', '10-Temp-0.99-Decay', 'mlrose', id=algo_ids)
    
    return [sa_nn_A, sa_nn_B, sa_nn_C]

def generate_ga_models(algo_ids=None):

    ga_nn_A = mlrose.NeuralNetwork(hidden_nodes = [100,20], activation = 'relu', \
                                algorithm = 'genetic_alg', max_iters = N_MAX_ITER, \
                                bias = True, is_classifier = True, learning_rate = 0.01, \
                                early_stopping = True, clip_max = 1, max_attempts = N_MAX_ATTEMPTS,
                                pop_size=50, mutation_prob=0.1)
    ga_nn_A = MachineLearningModel(ga_nn_A, 'GA-NN', '50-Pop-0.1-Mutation', 'mlrose', id=algo_ids)

    ga_nn_B = mlrose.NeuralNetwork(hidden_nodes = [100,20], activation = 'relu', \
                                algorithm = 'genetic_alg', max_iters = N_MAX_ITER, \
                                bias = True, is_classifier = True, learning_rate = 0.01, \
                                early_stopping = True, clip_max = 1, max_attempts = N_MAX_ATTEMPTS,
                                pop_size=50, mutation_prob=0.3)
    ga_nn_B = MachineLearningModel(ga_nn_B, 'GA-NN', '50-Pop-0.3-Mutation', 'mlrose', id=algo_ids)

    ga_nn_C = mlrose.NeuralNetwork(hidden_nodes = [100,20], activation = 'relu', \
                                algorithm = 'genetic_alg', max_iters = N_MAX_ITER, \
                                bias = True, is_classifier = True, learning_rate = 0.01, \
                                early_stopping = True, clip_max = 1, max_attempts = N_MAX_ATTEMPTS,
                                pop_size=500, mutation_prob=0.3)
    ga_nn_C = MachineLearningModel(ga_nn_C, 'GA-NN', '500-Pop-0.3-Mutation', 'mlrose', id=algo_ids)

    return [ga_nn_A, ga_nn_B, ga_nn_C]

#evaluation of randomized optimization model for neural network weights
def evaluate_model(algo, test_data, classes_list):
    '''
    using the fitted model, 
    '''
    print('evaluating model...')
    eval_start = time.time()
    predictions = algo.model.predict(test_data.data)
    eval_end = time.time()
    algo.set_evaluation_time(eval_end-eval_start)
    print('time to predict {} samples: {} seconds'.format(test_data.data.shape[0],algo.evaluation_time))
    
    cm = confusion_matrix(test_data.target, predictions)
    algo.cm = cm
    print('confusion matrix:\n',algo.cm)

    fp_rate, tp_rate, thresh = roc_curve(test_data.target, predictions)
    algo.precision = precision_score(test_data.target, predictions)
    algo.recall = recall_score(test_data.target, predictions)
    algo.f1 = f1_score(test_data.target, predictions)
    algo.roc_auc = roc_auc_score(test_data.target, predictions)
    algo.accuracy = accuracy_score(test_data.target,predictions)
    algo.balanced_accuracy = balanced_accuracy_score(test_data.target,predictions)
    print('precision         |', round(algo.precision,2))
    print('recall            |', round(algo.recall,2))
    print('F1                |', round(algo.f1,2))
    print('ROC-AUC           |', round(algo.roc_auc,2))
    print('accuracy          |', round(algo.accuracy,2))
    print('balanced accuracy |', round(algo.balanced_accuracy,2))
    return algo

def optimize_neural_net():
    algo_batch_id = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) #set ID for one run, so all the algos have the same ID

    # dictionary of model generator functions to iterate through training
    algo_dict = {
        'SA-NN':generate_sa_models,
        'RHC-NN':generate_rhc_models,
        'GA-NN':generate_ga_models
    }

    #load dataset
    if USE_DATASET == 'aps':
        dirty_train_df = pd.read_csv('data/aps/aps_failure_training_set.csv', na_values=['na'])
        dirty_test_df = pd.read_csv('data/aps/aps_failure_test_set.csv', na_values=['na'])
        print('using the dataset stored in ./data/aps')
        class_col = 'class'
    else:
        raise Exception('dataset other than `aps` was specified, no current implementation for other data sets.')

    #clean both datasets
    scaler = preprocessing.MinMaxScaler()
    train_df, test_df = clean_and_scale_dataset({'train':dirty_train_df, 'test':dirty_test_df}, scaler=scaler ,na_action=-1)
    #prep the datasets 
    [train_dataset, test_dataset], label_encoder = prep_data({'train':train_df, 'test':test_df}, shuffle_data=True, balance_method=BALANCE_METHOD, class_col=class_col)
    print('\nTRAINING DATA INFORMATION')
    print('{} maps to {}'.format(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print('size of training dataset:', train_dataset.data.shape)

    detail_df = pd.DataFrame(columns=['Model Name', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Accuracy', 'Training Time', 'Evaluation Time'])
    for algo_family_name, algo_generator in algo_dict.items():
        print('\ngenerating learning curves for {}'.format(algo_family_name))
        algo_list = algo_generator(algo_ids=algo_batch_id)
        for algo in algo_list:
            #first generate learning curves
            if PLOT_BATCH_LC:
                generate_learning_curves(algo, train_dataset, num_chunks=N_LC_CHUNKS, num_k=N_CV)

            print('\n\ntraining {} model on entire training dataset...'.format(algo.model_type))
            training_start = time.time()
            algo.model.fit(train_dataset.data, train_dataset.target)
            training_end = time.time()
            algo.set_training_time(training_end-training_start)
            print('time to train: {} seconds'.format(algo.training_time))
            print('algo loss', algo.model.loss)
            loss_list = algo.model.loss_list
            algo.iter_scores = loss_list

            algo = evaluate_model(algo, test_dataset, classes_list=label_encoder.classes_)
            detail_df = detail_df.append({'Model Name': algo.model_type, 
                                        'Precision': algo.precision, 
                                        'Recall':algo.recall, 
                                        'F1':algo.f1, 
                                        'ROC-AUC':algo.roc_auc, 
                                        'Accuracy':algo.accuracy, 
                                        'Training Time':algo.training_time,
                                        'Evaluation Time':algo.evaluation_time}, ignore_index=True)

        if PLOT_ACTION:
            if PLOT_BATCH_LC:
                plot_model_family_learning_curves(algo_family_name, 
                                                    algo_list, 
                                                    iter_based=False,
                                                    figure_action=PLOT_ACTION, 
                                                    figure_path='output/'+str(algo_batch_id)+'-neuralnet/figures/lc',
                                                    file_name=(str(algo_family_name)+'_batch'))

            plot_model_family_learning_curves(algo_family_name, 
                                                    algo_list, 
                                                    iter_based=True,
                                                    figure_action=PLOT_ACTION, 
                                                    figure_path='output/'+str(algo_batch_id)+'-neuralnet/figures/lc',
                                                    file_name=(str(algo_family_name)+'_iter'))

            plot_confusion_matrix(algo_family_name, 
                                    algo_list, 
                                    label_encoder.classes_, 
                                    figure_action=PLOT_ACTION, 
                                    figure_path='output/'+str(algo_batch_id)+'-neuralnet/figures/cm',
                                    file_name=(str(algo_family_name)))
    if OUTPUT_CSV:
        #only write results to .csv file if specified
        print('writing csv for model results')
        detail_df.to_csv('output/'+str(algo_batch_id)+'/models_summary_'+str(USE_DATASET)+'.csv', sep=',', encoding='utf-8', index=False)
 
# Section two of assignment, optimization of various problems via random search methods.
def generate_random_cities(n_cities=5):
    city_x = np.random.choice(range(n_cities), size=n_cities)
    city_y = np.random.choice(range(n_cities), size=n_cities)
    city_coords = [(city_x[i], city_y[i]) for i in range(n_cities)]
    return city_coords

class OptProbWrapper(object):
    '''
    wrapper class for optimization problems, solve each problem using all 4 optimization algorithms 
        with a single call to evaluate_algos().
    Also initializes the fitness function given a string input ('flipflop', 'fourpeaks', 'knapsack')
    '''
    def __init__(self, problem_name, gen_params=None, max_iter=1000, max_attempts=100, pop_size=100, restarts=0, keep_pct=0.2):
        self.problem_name = problem_name
        self.rhc = mlrose.algorithms.random_hill_climb
        self.sa = mlrose.algorithms.simulated_annealing
        self.ga = mlrose.algorithms.genetic_alg
        self.mimic = mlrose.algorithms.mimic
        self.max_attempts = max_attempts
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.restarts = restarts
        self.gen_params = gen_params
        self.keep_pct = keep_pct
        self.algo_result_dict = None
        self.rhc_train_time = None
        self.sa_train_time = None 
        self.ga_train_time = None 
        self.mimic_train_time = None

        if problem_name.lower() == 'flipflop':
            self.fitness_func = mlrose.FlipFlop()
        elif problem_name.lower() == 'fourpeaks':
            self.fitness_func = mlrose.FourPeaks()
        elif problem_name.lower() == 'knapsack':
            #weighs twice as much as it's worth
            weights = 2 * np.random.choice(range(1, self.gen_params+1, 1), size=self.gen_params)
            values = np.random.choice(range(1, self.gen_params+1, 1), size=self.gen_params)
            self.fitness_func = mlrose.Knapsack(weights, values)
        else:
            raise Exception('incorrect problem name provided, choose: knapsack, flipflop, or fourpeaks')

        self.problem = mlrose.opt_probs.DiscreteOpt(length = gen_params, fitness_fn = self.fitness_func, maximize=True)
            
    def evaluate_algos(self):
        #set max attempts for RHC and SA higher as complexity increases
        self.rhc_sa_max_attempts = int(self.max_attempts*(2**(self.gen_params/15))) if (self.gen_params/15)> 1 else self.max_attempts

        print('\nworking on problem:', self.problem_name, 'input size:', self.gen_params)
        start_time = time.time()
        #allow more more iterations without improvement when the parameters space is larger
        self.rhc_best_state, self.rhc_best_fitness, self.rhc_fitness_list, self.rhc_func_eval_count = self.rhc(self.problem, max_attempts=self.rhc_sa_max_attempts, max_iters=self.max_iter, restarts=self.restarts)
        self.rhc_train_time = (time.time()-start_time)
        if len(self.rhc_fitness_list) == self.max_iter:
            #this probably didn't converge then
            self.rhc_iter_to_converge = self.max_attempts
        else:
            self.rhc_iter_to_converge = len(self.rhc_fitness_list) - self.rhc_sa_max_attempts

        start_time = time.time()
        self.sa_best_state, self.sa_best_fitness, self.sa_fitness_list, self.sa_func_eval_count = self.sa(self.problem, schedule=mlrose.decay.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001), max_attempts=self.rhc_sa_max_attempts, max_iters=self.max_iter)
        self.sa_train_time = (time.time()-start_time)
        if len(self.sa_fitness_list) == self.max_iter:
            #this probably didn't converge then
            self.sa_iter_to_converge = self.max_attempts
        else:
            self.sa_iter_to_converge = len(self.sa_fitness_list) - self.rhc_sa_max_attempts
        
        start_time = time.time()
        self.ga_best_state, self.ga_best_fitness, self.ga_fitness_list, self.ga_func_eval_count = self.ga(self.problem, pop_size=self.pop_size, mutation_prob=0.3, max_attempts=self.max_attempts, max_iters=self.max_iter)
        self.ga_train_time = (time.time()-start_time)
        if len(self.ga_fitness_list) == self.max_iter:
            print()
            #this probably didn't converge then
            self.ga_iter_to_converge = self.max_attempts
        else:
            self.ga_iter_to_converge = len(self.ga_fitness_list) - self.max_attempts

        start_time = time.time()
        self.mimic_best_state, self.mimic_best_fitness, self.mimic_fitness_list, self.mimic_func_eval_count = self.mimic(self.problem, pop_size=self.pop_size, keep_pct=self.keep_pct, max_attempts=self.max_attempts, max_iters=self.max_iter)
        self.mimic_train_time = (time.time()-start_time)
        if len(self.mimic_fitness_list) == self.max_iter:
            #this probably didn't converge then
            self.mimic_iter_to_converge = self.max_attempts
        else:
            self.mimic_iter_to_converge = len(self.mimic_fitness_list) - self.max_attempts                    

    def get_best_fitness(self):
        self.best_fitness_dict = {key: value[1] for (key, value) in self.algo_result_dict.items()}
        print(self.best_fitness_dict)
        return max(self.best_fitness_dict, key=self.best_fitness_dict.get), max(self.best_fitness_dict.values())

    def get_algo_results(self, dict_key=None):
        if not self.algo_result_dict: #create if it's not created
            self.algo_result_dict = {
                'rhc':[self.rhc_best_state, self.rhc_best_fitness, self.rhc_fitness_list, self.rhc_func_eval_count, self.rhc_train_time, self.rhc_iter_to_converge],
                'sa':[self.sa_best_state, self.sa_best_fitness, self.sa_fitness_list, self.sa_func_eval_count, self.sa_train_time, self.sa_iter_to_converge],
                'ga':[self.ga_best_state, self.ga_best_fitness, self.ga_fitness_list, self.ga_func_eval_count, self.ga_train_time, self.ga_iter_to_converge],
                'mimic':[self.mimic_best_state, self.mimic_best_fitness, self.mimic_fitness_list, self.mimic_func_eval_count, self.mimic_train_time, self.mimic_iter_to_converge]
            }
        if dict_key:
            return self.algo_result_dict.get(dict_key, None)
        else:
            return self.algo_result_dict

# controller function for iterating through different optimization problems
def optimization_problems():
    problem_batch_id = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) #set ID for one run, so all the algos have the same ID
    folder_path = 'output/'+str(problem_batch_id)+'-optprobs'
    input_size_list = [10, 20, 30, 40, 50, 60, 70, 80]
    # fourpeaks - MIMIC, knapsack- GA, flipflop - SA
    problem_name_list = ['flipflop', 'knapsack', 'fourpeaks']

    detail_df = pd.DataFrame(columns=['problem_name', 'input_size', 'model_name', 'best_state', 'best_fitness', 'fitness_list', 'total_iter_count', 'func_eval_count', 'train_time', 'iter_to_converge', 'time_per_iter', 'approx_time_to_converge'])

    for problem_name in problem_name_list:
        for input_size in input_size_list:
            optimizer = OptProbWrapper(problem_name=problem_name, gen_params=input_size, max_iter=DO_MAX_ITER, max_attempts=DO_MAX_ATTEMPTS, pop_size=300, keep_pct=0.35)
            optimizer.evaluate_algos()
            results = optimizer.get_algo_results()

            for key, value_list in results.items():
                # problem_plotting_dict[key] = []
                detail_df = detail_df.append({'problem_name': problem_name,
                                                'input_size':input_size,
                                                'model_name': key,
                                                'best_state': value_list[0],
                                                'best_fitness': value_list[1],
                                                'fitness_list': value_list[2],
                                                'total_iter_count': len(value_list[2]),
                                                'func_eval_count': value_list[3],
                                                'train_time': value_list[4],
                                                'iter_to_converge': value_list[5],
                                                'time_per_iter': value_list[4]/len(value_list[2]),
                                                'approx_time_to_converge':(value_list[5]*(value_list[4]/len(value_list[2])))}, ignore_index=True)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # save the dataframe to CSV
    if OUTPUT_CSV:
        detail_df.to_csv(folder_path+'/optimization_details.csv', sep=',', encoding='utf-8', index=False)

    #save plots 
    if PLOT_ACTION:
        plot_opt_prob_curves(detail_df, plot_group='problem_name', line_group='model_name', x_col='input_size', y_col='iter_to_converge', figure_action=PLOT_ACTION, figure_path=folder_path, file_name='iters_to_converge')
        plot_opt_prob_curves(detail_df, plot_group='problem_name', line_group='model_name', x_col='input_size', y_col='approx_time_to_converge', figure_action=PLOT_ACTION, figure_path=folder_path, file_name='approx_time_to_converge')
        plot_opt_prob_curves(detail_df, plot_group='problem_name', line_group='model_name', x_col='input_size', y_col='best_fitness', figure_action=PLOT_ACTION, figure_path=folder_path, file_name='best_fitness')
        plot_opt_prob_curves(detail_df, plot_group='problem_name', line_group='model_name', x_col='input_size', y_col='func_eval_count', figure_action=PLOT_ACTION, figure_path=folder_path, file_name='func_eval_count')

    return None


def main():
    optimize_neural_net()
    optimization_problems()


if __name__ == '__main__':
    main()
