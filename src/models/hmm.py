import numpy as np

class HiddenMarkovModel:
    """Create an instance of the Hidden Markov Model to calculate probable hidden states
    """

    def __init__(self, 
                 observation_states: np.ndarray,
                 hidden_states: np.ndarray, 
                 prior_probabilities: np.ndarray, 
                 transition_probabilities: np.ndarray, 
                 emission_probabilities: np.ndarray):
        """Initialize HMM object

        Args:
            observation_states (np.ndarray): possible observation states in data
            hidden_states (np.ndarray): possible hidden states in data
            prior_probabilities (np.ndarray): start probabilities
            transition_probabilities (np.ndarray): array of transition probabilies
            emission_probabilities (np.ndarray): array of emission probabilities
        """        
        self.check_probability_matrices(observation_states,
                                        hidden_states,
                                        prior_probabilities,
                                        transition_probabilities, 
                                        emission_probabilities)

        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                  for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}
        

        self.prior_probabilities= prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities

    def check_probability_matrices(self, 
                                   observation_states: np.ndarray, 
                                   hidden_states: np.ndarray, 
                                   prior_probabilities: np.ndarray, 
                                   transition_probabilities: np.ndarray, 
                                   emission_probabilities: np.ndarray):
        """Check that the HMM arguments/probabilities are of expected dimensions

        Args:
            observation_states (np.ndarray): possible observation states in data
            hidden_states (np.ndarray): possible hidden states in data
            prior_probabilities (np.ndarray): start probabilities
            transition_probabilities (np.ndarray): transition probabilities
            emission_probabilities (np.ndarray): emission probabilities

        """        
        # get length of observation_states
        num_obs = len(observation_states)
        # get length of hidden_states
        num_hid = len(hidden_states)
        
        # Check length of prior probabilities, should have the same number as hidden_states
        if (len(prior_probabilities) != num_hid):
            raise ValueError ("Length of prior probabilities should match number of hidden states present")
        
        # Check that transition probabilites is a square matrix
        if (transition_probabilities.shape[0] != transition_probabilities.shape[1]):
            raise ValueError ("Transition probabilities should be a square matrix")
        # Check that transition probabilites have same number of dimensions as num_hid
        if (transition_probabilities.shape[0] != num_hid):
            raise ValueError ("Transition probabilities dimensions should match length of hidden states")
        # Check that the emission probabilities is of num_hid x num_obs
        if (emission_probabilities.shape[0] != num_hid):
            raise ValueError ("Emission probabilities should have rows matching number of hidden observations")
        if (emission_probabilities.shape[1] != num_obs):
            raise ValueError ("Emission probabilities should have columns matching number of observation states")