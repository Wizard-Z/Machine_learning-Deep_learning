import pickle
import matplotlib.pyplot as plt
from train_utils import *
from utils import read_pickle
def model(data, ix_to_char, char_to_ix, num_iterations = 10000, n_a = 500, sample_no = 10, vocab_size = 27):
    """
    Train the model
 
    
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    sample_no -- number of names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text, size of the vocabulary
    
    Returns:
    parameters -- learned parameters
    """

    # Retrieve n_x and n_y from vocab_size
    iter_loss = []
    n_x, n_y = vocab_size, vocab_size
    
    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
    loss = get_initial_loss(vocab_size, sample_no)
    
    # Build list of all names (training examples).

    examples = data.split('\n')
    
    # Shuffle list of all  names
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))
    
    # Optimization loop
    for j in range(num_iterations):     
        # Use the hint above to define one training example (X,Y) (≈ 2 lines)
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]] 
        Y = X[1:] + [char_to_ix["\n"]]
        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
        
        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j% 100 == 0:
          iter_loss.append(loss)
        if j % 2000 == 0:
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # The number of names to print
            seed = 0
            for name in range(sample_no):
                
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                
                seed += 1  # increment the seed by one. 
      
            print('\n')
        
    return (parameters,iter_loss)

def main():
    # dataset
    print('Default: Training for Boys names.\nTo change edit train.py!\t(Change pickle file) \n')
    data = read_pickle('datasets/boys_string.pickle')[0]
    ix_to_char,char_to_ix = read_pickle('datasets/ix_char_ix.pickle')[0]
    print('Training Data Loaded')
    num_iterations = int(input('Enter Number of Iteration.\n(Recommended: 2000+) '))
    parameters, iter_loss = model(data, ix_to_char, char_to_ix, num_iterations)
    ch = input('Want to Save Parameters and loss (Y/n) : ').lower()
    if ch == 'y':
        with open('datasets/n_parameters.pickle', 'wb') as file:
            pickle.dump(parameters, file)
        with open('datasets/n_loss.pickle', 'wb') as file:
            pickle.dump(iter_loss, file)
        print('Files Saved!')
    plt.plot(iter_loss)
    plt.xlabel('Iteration in 100\'s ')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()
    print('Training Done!')


if __name__ == '__main__':
    main()
