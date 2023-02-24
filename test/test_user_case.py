"""
UCSF BMI203: Biocomputing Algorithms
Author: Han Chen   
Date: 2023-02-24
Program: models
Description: Test user cases for HMM VITERBI
"""
import pytest
import numpy as np
from models.hmm import HiddenMarkovModel
from models.decoders import ViterbiAlgorithm


def test_use_case_lecture():
    """Test case re: funding source and student happiness
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_one():
    """Test case re: traffic conditions
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    # Update hidden states to fix error
    new_hidden_states = np.array(['no-traffic', 'no-traffic', 'traffic', 'traffic', 'traffic', 'no-traffic'])
    assert np.alltrue(use_case_decoded_hidden_states == new_hidden_states)


def test_user_case_two():
    """Test case re: read books or not
    Hypothesis being that if it's a fiction book, it's more likely to have been read. 
    """
    observation_states = ['read','unread'] 
    hidden_states = ['fiction','nonfiction'] 

    prior_probs = np.array([0.6, 0.4])
    transition_probs = np.array([0.6, 0.4, 0.7, 0.3]).reshape(2,2)
    emission_probs = np.array([0.8, 0.2, 0.4, 0.6]).reshape(2,2)

    hmm_fiction_book = HiddenMarkovModel(observation_states = observation_states,
                                            hidden_states = hidden_states,
                                            prior_probabilities = prior_probs,
                                            transition_probabilities = transition_probs,
                                            emission_probabilities = emission_probs)

    fiction_viterbi = ViterbiAlgorithm(hmm_fiction_book)

    observed_states = ['read', 'read', 'unread', 'read', 'unread', 'unread', 'unread', 'read']

    pred_seq = fiction_viterbi.best_hidden_state_sequence(observed_states)
    seq = ['fiction', 'fiction', 'nonfiction', 'fiction', 'nonfiction', 'fiction', 'nonfiction', 'fiction']
    assert np.alltrue(pred_seq == seq)


def test_user_case_three():
    """Testing improper inputs
    """
    # testing improper number of prior probabilites
    observation_states1 = ['obs1', 'obs2', 'obs3']
    hidden_states1 = ['hid1','hid2'] 

    prior_probs1 = np.array([0.6, 0.4, 0.2])
    transition_probs1 = np.array([0.6, 0.4, 0.7, 0.3]).reshape(2,2)
    emission_probs1 = np.array([0.8, 0.2, 0.4, 0.6]).reshape(2,2)
    with pytest.raises(ValueError, match = "Length of prior probabilities should match number of hidden states present"):
        HiddenMarkovModel(observation_states = observation_states1,
                            hidden_states = hidden_states1,
                            prior_probabilities = prior_probs1,
                            transition_probabilities = transition_probs1,
                            emission_probabilities = emission_probs1)
        
    # Testing non square matrix
    observation_states2 = ['obs1', 'obs2', 'obs3']
    hidden_states2 = ['hid1','hid2'] 

    prior_probs2 = np.array([0.6, 0.4])
    transition_probs2 = np.array([0.6, 0.4, 0.7, 0.3, 0.8, 0.3]).reshape(3,2)
    emission_probs2 = np.array([0.8, 0.2, 0.4, 0.6]).reshape(2,2)
    with pytest.raises(ValueError, match = "Transition probabilities should be a square matrix"):
        HiddenMarkovModel(observation_states = observation_states2,
                            hidden_states = hidden_states2,
                            prior_probabilities = prior_probs2,
                            transition_probabilities = transition_probs2,
                            emission_probabilities = emission_probs2)
        

    # Testing dimensions not matching the hidden states
    observation_states3 = ['obs1', 'obs2', 'obs3']
    hidden_states3 = ['hid1','hid2', 'hid3'] 

    prior_probs3 = np.array([0.6, 0.4, 0.2])
    transition_probs3 = np.array([0.6, 0.4, 0.7, 0.3]).reshape(2,2)
    emission_probs3 = np.array([0.8, 0.2, 0.4, 0.6]).reshape(2,2)
    with pytest.raises(ValueError, match = "Transition probabilities dimensions should match length of hidden states"):
        HiddenMarkovModel(observation_states = observation_states3,
                            hidden_states = hidden_states3,
                            prior_probabilities = prior_probs3,
                            transition_probabilities = transition_probs3,
                            emission_probabilities = emission_probs3)
        

    # Testing emission probs
    observation_states4 = ['obs1', 'obs2', 'obs3']
    hidden_states4 = ['hid1','hid2'] 

    prior_probs4 = np.array([0.6, 0.4])
    transition_probs4 = np.array([0.6, 0.4, 0.7, 0.3]).reshape(2,2)
    emission_probs4 = np.array([0.8, 0.2, 0.4, 0.6, 0.8, 0.2]).reshape(3,2)
    with pytest.raises(ValueError, match = "Emission probabilities should have rows matching number of hidden observations"):
        HiddenMarkovModel(observation_states = observation_states4,
                            hidden_states = hidden_states4,
                            prior_probabilities = prior_probs4,
                            transition_probabilities = transition_probs4,
                            emission_probabilities = emission_probs4)
        
    observation_states5 = ['obs1', 'obs2', 'obs3']
    hidden_states5 = ['hid1','hid2'] 

    prior_probs5 = np.array([0.6, 0.4])
    transition_probs5 = np.array([0.6, 0.4, 0.7, 0.3]).reshape(2,2)
    emission_probs5 = np.array([0.8, 0.2, 0.4, 0.6]).reshape(2,2)
    with pytest.raises(ValueError, match = "Emission probabilities should have columns matching number of observation states"):
        HiddenMarkovModel(observation_states = observation_states5,
                            hidden_states = hidden_states5,
                            prior_probabilities = prior_probs5,
                            transition_probabilities = transition_probs5,
                            emission_probabilities = emission_probs5)
