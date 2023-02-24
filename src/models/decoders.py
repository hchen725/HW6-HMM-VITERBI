import copy
import numpy as np

class ViterbiAlgorithm:
    """_summary_
    """    

    def __init__(self, hmm_object):
        """_summary_

        Args:
            hmm_object (_type_): _description_
        """              
        self.hmm_object = hmm_object
    
    def trellis_node_step (self, prev_delta, prob_emiss, prob_trans, current_observation):
        # Multiply previous delta with transition probabilities 
        _delta_trans = np.multiply(prev_delta, prob_trans.T)
        # Get the likely scenario of the node
        probable_state = np.argmax(_delta_trans, axis = 1)
        probable_state_probs = np.amax(_delta_trans, axis = 1)
        # Calculate the new delta
        new_delta = probable_state_probs * prob_emiss[:, current_observation]
        # scale delta 
        new_delta = new_delta / np.sum(new_delta)

        return new_delta, probable_state

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            decode_observation_states (np.ndarray): _description_

        Returns:
            np.ndarray: _description_s
        """        
        
        # Pull out hmm object for easier ref
        observation_states = self.hmm_object.observation_states
        observation_states_dict = self.hmm_object.observation_states_dict
        hidden_states = self.hmm_object.hidden_states
        hidden_states_dict = self.hmm_object.hidden_states_dict
        prior_probs = self.hmm_object.prior_probabilities
        transition_probs = self.hmm_object.transition_probabilities
        emission_probs = self.hmm_object.emission_probabilities

        # # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        # path = np.zeros((len(decode_observation_states), 
        #                  len(hidden_states)))
        # path[0,:] = [hidden_state_index for hidden_state_index in range(len(hidden_states))]

        # best_path = np.zeros((len(decode_observation_states), 
        #                       len(hidden_states)))        
        
        # Initialize to hold paths of previous states:
        all_prev_states = np.zeros((len(decode_observation_states), 
                                    len(hidden_states)))
        all_prev_states[0,:] = [hidden_state_index for hidden_state_index in range(len(hidden_states))]

        # calculate initial delta
        delta = np.multiply(prior_probs, emission_probs[:,observation_states_dict[decode_observation_states[0]]])
        # Scale
        delta = delta / np.sum(delta)
        # Loop through each observation state node:
        for trellis_node in range(1, len(decode_observation_states)): 
             # Get the current observation state
            current_observation = observation_states_dict[decode_observation_states[trellis_node]]
            # Calculate state and update delta
            delta, prev_states = self.trellis_node_step(delta, emission_probs, transition_probs, current_observation)
            # Store states
            all_prev_states[trellis_node, :] = prev_states
        
        # Back trace to get highest probability sequence
        state = np.argmax(delta)
        #sequence_prob = np.amax(delta)
        best_path = []
        for prev_states in all_prev_states[::-1]:
            state = prev_states[int(state)]
            best_path.append(state)
        
        hidden_state_path = np.array([hidden_states_dict[i] for i in best_path[::-1]])
        # return hidden_state_path, sequence_prob
        return hidden_state_path


        # # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        # for trellis_node in range(1, len(decode_observation_states)):

        #     # TODO: comment the initialization, recursion, and termination steps

        #     product_of_delta_and_transition_emission =  np.multiply()
            
        #     # Update delta and scale

        #     # Select the hidden state sequence with the maximum probability

        #     # Update best path
        #     for hidden_state in range(len(self.hmm_object.hidden_states)):
            
        #     # Set best hidden state sequence in the best_path np.ndarray THEN copy the best_path to path

        #     #path = best_path.copy()

        # # Select the last hidden state, given the best path (i.e., maximum probability)

        # best_hidden_state_path = np.array([])
        # best hidden state path is the most likely hidden state sequence
        # return best_hidden_state_path
    