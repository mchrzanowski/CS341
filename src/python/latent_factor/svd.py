import numpy
import pickle

from math import sqrt


def plot_errors_per_k_and_lambda(lambdas, errors, error_type, graph_label):
    import matplotlib.pyplot as plt

    min_x = max_x = min_y = max_y = None

    for lambda_weight in lambdas:
        x_points = tuple(x[0] for x in errors[lambda_weight][error_type])
        y_points = tuple(x[1] for x in errors[lambda_weight][error_type])

        if min_x == None:
            min_x = min(x_points)
            max_x = max(x_points)

            min_y = min(y_points)
            max_y = max(y_points)
        else:
            if min(y_points) < min_y:
                min_y = min(y_points)
            if max(y_points) > max_y:
                max_y = max(y_points)

        plt.plot(x_points, y_points, label="Lambda=%s" % str(lambda_weight),
            marker='o')

    plt.xlabel("K")
    plt.ylabel("Error")
    plt.legend(loc=0)
    plt.axis([min_x, max_x, min_y - 1000, max_y + 1000])
    plt.title(graph_label + " Error")
    plt.show()


def plot_errors_per_iteration(errors):
    import matplotlib.pyplot as plt

    x_points = tuple(x[0] for x in errors)
    y_points = tuple(x[1] for x in errors)
    
    # line of code is very close to the documentation for matplotlib as to
    # how to use plt.plot:
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    plt.plot(x_points, y_points, color='green', linestyle='dashed', marker='o',
        markerfacecolor='red', markersize=8)
    
    plt.xlabel("Iteration")
    plt.ylabel("Error")

    plt.axis([min(x_points), max(x_points), min(y_points) - 1000, max(y_points) + 1000])
    plt.title("Error as a function of Iteration")
    plt.show()


def test(r_filename, Q, P, user_biases, movie_biases, mean_rating, movies, users, **garbage):
    E = 0
    with open(r_filename, 'rb') as f:
        for line in f:
            user, movie, rating = line.split()
            if user not in users or movie not in movies:
                continue
            E += (float(rating) - mean_rating - user_biases[users[user]] - movie_biases[movies[movie]] - 
                numpy.dot(Q[movies[movie]], P[users[user]])) ** 2

    return E


def train(r_filename, k, eta, lambda_weight, iterations, verbose=False):
    if verbose:
        print "Training. Lambda: %s, Eta: %s, Iterations: %s, k: %s" % (lambda_weight, eta, iterations, k)

    # first, compute m & n by creating maps from the input elements to matrix indices.
    movies = dict()
    users = dict()
    ratings_seen_so_far = 0
    rating_accumulator = 0

    user_input = dict()
    movie_input = dict()
    ratings_input = dict()

    with open(r_filename, 'rb') as f:
        for i, line in enumerate(f):
            user, movie, rating = line.split()
            user_input[i] = user
            movie_input[i] = movie
            ratings_input[i] = rating

            if user not in users:
                users[user] = len(users)
            if movie not in movies:
                movies[movie] = len(movies)

            rating_accumulator += float(rating)
            ratings_seen_so_far += 1

    mean_rating = float(rating_accumulator) / ratings_seen_so_far
    if verbose:
        print "Mean rating: %s" % mean_rating

    # initialize Q & P
    Q = sqrt((5. - mean_rating) / k) * numpy.random.rand(len(movies), k)   # Q = movies X k
    P = sqrt((5. - mean_rating) / k) * numpy.random.rand(len(users), k)    # P = users X k

    # initialize bias terms:
    user_biases = -mean_rating + 5 * numpy.random.rand(len(users))
    movie_biases = -mean_rating + 5 * numpy.random.rand(len(movies))  

    errors = []
    for iteration in xrange(1, iterations + 1):

        for i in user_input:
            user = user_input[i]
            movie = movie_input[i]
            rating = ratings_input[i]

            error = float(rating) - mean_rating - user_biases[users[user]] - \
                movie_biases[movies[movie]] - numpy.dot(Q[movies[movie]], P[users[user]])

            q_row_updated = Q[movies[movie]] + eta * (error * P[users[user]] - lambda_weight * Q[movies[movie]])
            p_row_updated = P[users[user]] + eta * (error * Q[movies[movie]] - lambda_weight * P[users[user]])
            user_biases_updated = user_biases[users[user]] + eta * (error - lambda_weight * user_biases[users[user]])
            movie_biases_updated = movie_biases[movies[movie]] + eta * (error - lambda_weight * movie_biases[movies[movie]])

            Q[movies[movie]] = q_row_updated
            P[users[user]] = p_row_updated
            user_biases[users[user]] = user_biases_updated
            movie_biases[movies[movie]] = movie_biases_updated

        # error estimate
        E = 0
        for i in user_input:
            user = user_input[i]
            movie = movie_input[i]
            rating = ratings_input[i]
            E += (float(rating) - mean_rating - user_biases[users[user]] - \
                movie_biases[movies[movie]] - numpy.dot(Q[movies[movie]], P[users[user]])) ** 2

        # regularization
        for user in users:
            E += lambda_weight * numpy.dot(P[users[user]], P[users[user]])
            E += lambda_weight * user_biases[users[user]] ** 2
        for movie in movies:
            E += lambda_weight * numpy.dot(Q[movies[movie]], Q[movies[movie]])
            E += lambda_weight * movie_biases[movies[movie]] ** 2

        if verbose:
            print "%s : %s" % (iteration, E)
        
        errors.append((iteration, E))

    return {'errors' : errors, 'Q': Q, 'P' : P, 'user_biases' : user_biases,
        'movie_biases' : movie_biases, 'mean_rating' : mean_rating, 'movies' : movies, 'users' : users}


def main(train_file, test_file, part_to_run, use_cached_data=False, plot=False, verbose=False):
    ITERATIONS = 20
    MAX_K = 10

    if part_to_run.upper() == "D1":

        print "Running part D, part B"
        eta = 0.03
        
        data = train(train_file, 20, eta, 0.2, ITERATIONS, verbose)
        print test(test_file, **data)
        if plot:
            plot_errors_per_iteration(data['errors'])

    elif part_to_run.upper() == "D2":
                
        print "Running part D, part C"
        pickling_file = "./error_data_part_d"

        K = tuple(i for i in xrange(1, MAX_K + 1))
        lambdas = (0.0, 0.2)
        eta = 0.03
        if not use_cached_data:
            resulting_errors = dict()
            for lambda_weight in lambdas:
                resulting_errors[lambda_weight] = dict()
                resulting_errors[lambda_weight]['TRAIN'] = list()
                resulting_errors[lambda_weight]['TEST'] = list()
                for k in K:
                    data = train(train_file, k, eta, lambda_weight, ITERATIONS, verbose)
                    E_train = test(train_file, **data) # this requires coordination of the return dict and function args!
                    E_test = test(test_file, **data)
                    resulting_errors[lambda_weight]['TRAIN'].append((k, E_train))
                    resulting_errors[lambda_weight]['TEST'].append((k, E_test))
                    # pickle data constantly!
                    pickle.dump(resulting_errors, open(pickling_file, 'wb'))
        else:
            resulting_errors = pickle.load(open(pickling_file, 'rb'))
        
        if plot:
            plot_errors_per_k_and_lambda(lambdas, resulting_errors, 'TRAIN', "Training")
            plot_errors_per_k_and_lambda(lambdas, resulting_errors, 'TEST', "Testing")


if __name__ == "__main__":
    import argparse
    import time
    
    start_time = time.time()
    parser = parser = argparse.ArgumentParser(description="HW3Q1")
    parser.add_argument('-part', required=True, type=str, help="Portion of Q1 to run")
    parser.add_argument('-l', required=False, action='store_true', help="Use cached data.")
    parser.add_argument('-plot', required=False, action='store_true', help="Show plots.")
    parser.add_argument('-verbose', required=False, action='store_true', help="Verbosity.")
    parser.add_argument('-tr', required=True, type=str, help="Train file.")
    parser.add_argument('-te', required=True, type=str, help="Test file.")

    args = vars(parser.parse_args())
    
    main(args['tr'], args['te'], args['part'], args['l'], args['plot'], args['verbose'])
    end_time = time.time()
    print "Runtime: %s seconds" % (end_time - start_time)
