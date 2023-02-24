import copy
import numpy as np

class ViterbiAlgorithm:
    """Implementation of the Viterbi Algorithm
    """    

    def __init__(self, hmm_object):
        """Initialization of the Viterbi Algorithm

        Args:
            hmm_object (hmm_object): HMM object created from initializing HiddenMarkovModel
        """        
        self.hmm_object = hmm_object
    

    def best_hidden_state_sequence(self, 
                                   decode_observation_states: np.ndarray) -> np.ndarray:
        """Find the hidden state sequence for a set of observation states

        Args:
            decode_observation_states (np.ndarray): Observation states to decode

        Returns:
            hidden_state_path (np.ndarray): Most probable sequence of of hidden states 
        """        
        
        # Pull out hmm object for easier ref
        observation_states = self.hmm_object.observation_states
        observation_states_dict = self.hmm_object.observation_states_dict
        hidden_states = self.hmm_object.hidden_states
        hidden_states_dict = self.hmm_object.hidden_states_dict
        prior_probs = self.hmm_object.prior_probabilities
        transition_probs = self.hmm_object.transition_probabilities
        emission_probs = self.hmm_object.emission_probabilities

        # Get the number of possible hidden states
        num_hidden = len(hidden_states)
        # Get the number of observations to decode
        num_obs = len(decode_observation_states) 

        # Initialize to store paths
        paths = np.zeros((num_hidden, num_obs))
        paths[:,0] = [hidden_state_index for hidden_state_index in range(len(hidden_states))]

        # Initialize to store all deltas that are calculated
        all_deltas = np.zeros((num_hidden, num_obs)) # all deltas
        all_deltas[:, 0] = np.multiply(prior_probs, emission_probs[:, observation_states_dict[decode_observation_states[0]]])

        # Iterate through the node
        for trellis_node in range(1, num_obs):
            # Get the current observation state
            current_observation_state = observation_states_dict[decode_observation_states[trellis_node]]
            prev_delta = all_deltas[:, trellis_node-1]
            for _hidden_state in range(0, num_hidden):
                # Multiple previous delta
                delta_trans = np.multiply(transition_probs[:, _hidden_state], prev_delta)
                # Update path with most probable delta
                paths[_hidden_state, trellis_node] = np.argmax(delta_trans)
                # Compute new delta with the current observation satte
                new_delta = np.max(delta_trans) * emission_probs[_hidden_state, current_observation_state]
                # Update deltas
                all_deltas[_hidden_state, trellis_node] = new_delta
                
        
        # Back trace to get the best path
        best_path = np.zeros(num_obs)
        # Get the index of the final delta
        best_path[num_obs-1] = np.argmax(all_deltas[:, num_obs-1])
        for n in range(num_obs-1, 0, -1):
            best_path[n-1] = paths[int(best_path[n]), n]

        # Convert best_path into words
        hidden_state_path = np.array([hidden_states_dict[i] for i in best_path])
        return hidden_state_path