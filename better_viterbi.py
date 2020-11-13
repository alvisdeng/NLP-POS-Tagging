'''
Yihui Peng
Ze Xuan Ong
Jocelyn Huang
Noah A. Smith
Yifan Xu

Usage: python viterbi.py <HMM_FILE> <TEXT_FILE> <OUTPUT_FILE>

Apart from writing the output to a file, the program also prints
the number of text lines read and processed, and the time taken
for the entire program to run in seconds. This may be useful to
let you know how much time you have to get a coffee in subsequent
iterations.
'''

import math
import sys
import time
import numpy as np
from collections import defaultdict

# Magic strings and numbers
HMM_FILE = sys.argv[1]
OLD_HMM_FILE = sys.argv[2]
TEXT_FILE = sys.argv[3]
OUTPUT_FILE = sys.argv[4]
TRANSITION_TAG = "trans"
EMISSION_TAG = "emit"
OOV_WORD = "OOV"         # check that the HMM file uses this same string
INIT_STATE = "init"      # check that the HMM file uses this same string
FINAL_STATE = "final"    # check that the HMM file uses this same string


class Viterbi():
    def __init__(self):
        # transition and emission probabilities. Remember that we're not dealing with smoothing 
        # here. So for the probability of transition and emission of tokens/tags that we haven't 
        # seen in the training set, we ignore thm by setting the probability an impossible value 
        # of 1.0 (1.0 is impossible because we're in log space)

        self.transition = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.emission = defaultdict(lambda: defaultdict(lambda: 1.0))

        self.old_transition = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.old_emission = defaultdict(lambda: defaultdict(lambda: 1.0))
        # keep track of states to iterate over 
        self.states = set()
        self.POSStates = set()
        # store vocab to check for OOV words
        self.vocab = set()

        # text to run viterbi with
        self.text_file_lines = []
        with open(TEXT_FILE, "r") as f:
            self.text_file_lines = f.readlines()

    def readModel(self):
        # Read HMM transition and emission probabilities
        # Probabilities are converted into LOG SPACE!
        with open(HMM_FILE, "r") as f:
            for line in f:
                line = line.split()

                # Read transition
                # Example line: trans NN NNPS 9.026968067100463e-05
                # Read in states as prev_state -> state
                if line[0] == TRANSITION_TAG:
                    prev_state = line[1],line[2]
                    state,trans_prob = line[3:5]
                    self.transition[prev_state][state] = math.log(float(trans_prob))
                    self.states.add(prev_state[0])
                    self.states.add(prev_state[1])
                    self.states.add(state)

                # Read in states as state -> word
                elif line[0] == EMISSION_TAG:
                    (state, word, emit_prob) = line[1:4]
                    self.emission[state][word] = math.log(float(emit_prob))
                    self.states.add(state)
                    self.vocab.add(word)
        with open(OLD_HMM_FILE,"r") as f:
            for line in f:
                line = line.split()
                if line[0] == TRANSITION_TAG:
                    (prev_state, state, trans_prob) = line[1:4]
                    self.old_transition[prev_state][state] = math.log(float(trans_prob))
                elif line[0] == EMISSION_TAG:
                    (state, word, emit_prob) = line[1:4]
                    self.old_emission[state][word] = math.log(float(emit_prob))

        # Keep track of the non-initial and non-final states
        self.POSStates = self.states.copy()
        self.POSStates.remove(INIT_STATE)
        self.POSStates.remove(FINAL_STATE)


    # run Viterbi algorithm and write the output to the output file
    def runViterbi(self):
        result = []
        for line in self.text_file_lines:
            result.append(self.viterbiLine(line))

        # Print output to file
        with open(OUTPUT_FILE, "w") as f:
            for line in result:
                f.write(line)
                f.write("\n")

    # TODO: Implement this

    # run Viterbi algorithm on a line of text 
    # Input: A string representing a sequence of tokens separated by white spaces 
    # Output: A string representing a sequence of POS tags.

    # Things to keep in mind:
    # 1. Probability calculations are done in log space. 
    # 2. Ignore smoothing in this case. For  probabilities of emissions that we haven't seen
    # or  probabilities of transitions that we haven't seen, ignore them. (How to detect them?
    # Remember that values of self.transition and self.emission are default dicts with default 
    # value 1.0!)
    # 3. A word is treated as an OOV word if it has not been seen in the training set. Notice 
    # that an unseen token and an unseen transition/emission are different things. You don't 
    # have to do any additional thing to handle OOV words.
    # 4. There might be cases where your algorithm cannot proceed. (For example, you reach a 
    #     state that for all prevstate, the transition probability prevstate->state is unseen)
    #     Just return an empty string in this case. 
    # 5. You can set up your Viterbi matrix (but we all know it's better to implement it with a 
    #     python dictionary amirite) in many ways. For example, you will want to keep track of 
    #     the best prevstate that leads to the current state in order to backtrack and find the 
    #     best sequence of pos tags. Do you keep track of it in V or do you keep track of it 
    #     separately? Up to you!
    # 6. Don't forget to handle final state!
    # 7. If you are reading this during spring break, mayyyyybe consider taking a break from NLP 
    # for a bit ;)

    def viterbiLine(self, line):
        words = line.split()
        states = list(self.POSStates)

        rows = len(words)
        cols = len(states)

        # TODO: Initialize DP matrix for Viterbi here
        dp_tab = np.zeros([rows,cols])
        dp_tab.fill(-np.inf)
        path = np.zeros([rows,cols])
        dp_tab.fill(-np.inf)

        first_word = OOV_WORD if words[0] not in self.vocab else words[0]
        valid_idx = []
        for i,state in enumerate(states):
            if self.transition[INIT_STATE,INIT_STATE][state] != 1 and self.emission[state][first_word] != 1:
                valid_idx.append(i)
                trans_prob = self.transition[INIT_STATE,INIT_STATE][state]
                emis_prob = self.emission[state][first_word]
                dp_tab[0][i] = trans_prob + emis_prob
            elif self.old_transition[INIT_STATE][state] != 1 and self.old_emission[state][first_word] != 1:
                valid_idx.append(i)
                trans_prob = self.old_transition[INIT_STATE][state]
                emis_prob = self.old_emission[state][first_word]
                dp_tab[0][i] = trans_prob + emis_prob

            path[0][i] = i

        for (i, word) in enumerate(words):
            # replace unseen words as oov
            if word not in self.vocab:
                word = OOV_WORD
            if i == 0:
                continue

            # TODO: Fill up your DP matrix
            max_probs = []
            for j, state in enumerate(states):
                probs = {}
                for k in valid_idx:
                    if i == 1:
                        prev_state = INIT_STATE,states[k]
                    else:
                        prev_state = states[int(path[i-1][k])],states[k]

                    if self.transition[prev_state][state] != 1 and self.emission[state][word] != 1:
                        trans_prob = self.transition[prev_state][state]
                        emis_prob = self.emission[state][word]
                        prob = dp_tab[i-1][k] + trans_prob + emis_prob
                    elif self.old_transition[prev_state[1]][state] != 1 and self.old_emission[state][word] != 1:
                        trans_prob = self.old_transition[prev_state[1]][state]
                        emis_prob = self.old_emission[state][word]
                        prob = dp_tab[i-1][k] + trans_prob + emis_prob
                    else:
                        prob = -np.inf
                    probs[prob] = k

                if len(probs) != 0:
                    max_prob = max(probs)
                    max_probs.append(max_prob)
                    path[i][j] = probs[max_prob]
                else:
                    return ''
            
            valid_idx = []
            for o in range(len(max_probs)):
                if max_probs[o] > -np.inf:
                    valid_idx.append(o)
            dp_tab[i] = np.array(max_probs)

        # TODO: Handle best final state
        final_probs = {}
        for k in valid_idx:
            prev_state = states[k]
            trans_prob = self.transition[state][FINAL_STATE] if self.transition[state][FINAL_STATE] != 1 else 0
            prob = dp_tab[-1][k] + trans_prob
            final_probs[prob] = k
        final_idx = final_probs[max(final_probs)]

        # TODO: Backtrack and find the optimal sequence. 
        tags_idx_reverse = [final_idx]
        for i in range(len(words)-1,0,-1):
            tags_idx_reverse.append(int(path[i][final_idx]))
            final_idx = int(path[i][final_idx])
        tags = []
        for idx in tags_idx_reverse[::-1]:
            tags.append(states[idx])
        return " ".join(tags)

if __name__ == "__main__":
    # Mark start time
    t0 = time.time()
    viterbi = Viterbi()
    viterbi.readModel()
    viterbi.runViterbi()
    # Mark end time
    t1 = time.time()
    print("Time taken to run: {}".format(t1 - t0))

