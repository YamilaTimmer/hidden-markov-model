import math
import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product
from math import exp, log

class HiddenMarkovModel:
    def __init__(self, startprob, transprob, emissionprob, states, emissions):
        self.startprob = startprob
        self.transprob = transprob
        self.emissionprob = emissionprob
        self.states = states
        self.emissions = emissions

        self.n_components = self.startprob.shape[0]
        self.n_observations = len(self.emissionprob[1])


    def __repr__(self):
        return (f"HiddenMarkovModel(n_components={self.n_components}, "
                f"n_observations={self.n_observations})")


    def __str__(self):
        return (
            "Hidden Markov Model\n"
            f"Start probabilities:\n{self.startprob}\n\n"
            f"Transition matrix:\n{self.transprob}\n\n"
            f"Emission probabilities:\n{self.emissionprob}\n"
        )


    def sample(self, n_steps):
        resulting_emissions = np.array([], dtype=int)
        resulting_states = np.array([], dtype=int)

        # Bepaal start state
        start_state = random.choices(self.states, weights=self.startprob)
        resulting_states = np.append(resulting_states, start_state)

        for i in range(n_steps-1):
            resulting_emissions = np.append(resulting_emissions, random.choices(self.emissions, weights=self.emissionprob[resulting_states[-1]])[0])
            resulting_states = np.append(resulting_states, random.choices(self.states, weights=self.transprob[resulting_states[-1]])[0])

        # Voeg laatste emissie toe
        resulting_emissions = np.append(resulting_emissions, random.choices(self.emissions, weights=self.emissionprob[resulting_states[-1]])[0])

        return resulting_states, resulting_emissions


    def determine_observations(self, states_sequence, emissions_sequence):

        observed_transitions = np.zeros((3, 3))
        observed_emissions = np.zeros((3, 4))


        for i in range(1, len(states_sequence) - 1):
            observed_transitions[states_sequence[i], states_sequence[i + 1]] += 1

        if len(states_sequence) > 1:
            for i in range(len(emissions_sequence)):
                observed_emissions[states_sequence[i], emissions_sequence[i]] += 1

        for i in range(observed_transitions.shape[0]):
            row_sum = observed_transitions[i].sum()
            observed_transitions[i] /= row_sum

        for i in range(observed_emissions.shape[0]):
            row_sum = observed_emissions[i].sum()
            observed_emissions[i] /= row_sum

        return observed_transitions, observed_emissions


    def score(self, emission_sequence, state_sequence=None):
        if state_sequence is None:
        # Implementatie forward-algoritme
            return

        else:
            # klein getal om 0 waarden mee te vervangen voor de log berekening
            epsilon = 1e-10

            log_probability = (
                    math.log(self.startprob[state_sequence[0]]) +
                    # als emissiekans gelijk is aan 0, wordt epsilon gebruikt
                    math.log(max(self.emissionprob[state_sequence[0]][emission_sequence[0]], epsilon))
            )

            for i in range(1, len(state_sequence)):
                log_probability += (
                    math.log(self.transprob[state_sequence[i - 1]][state_sequence[i]]) +
                    math.log(max(self.emissionprob[state_sequence[i]][emission_sequence[i]], epsilon))
                    )

        return log_probability


    def predict(self, emission_sequence):
        num_observations = len(emission_sequence)
        num_states = len(self.states)
        probability_table = np.zeros((num_states, num_observations))
        best_prev = np.zeros((num_states, num_observations))

        # Eerste waarneming
        for s in self.states:
            probability_table[s, 0] = self.startprob[s] * self.emissionprob[s][emission_sequence[0]]

        # Verdere waarnemingen
        for t in range(1, num_observations):
            for s in self.states:
                probs = [probability_table[p, t - 1] * self.transprob[p][s] for p in self.states]
                best_prev[s, t] = np.argmax(probs)
                probability_table[s, t] = max(probs) * self.emissionprob[s][emission_sequence[t]]

        path = [0] * num_observations
        path[1] = int(np.argmax(probability_table[:, -1]))

        for t in range (num_observations - 1, 0, -1):
            path[t-1] = int(best_prev[path[t], t])
        
        return path


# Helper functies voor visualiseren plots
def bar_plot(states, emissions, quantity_data_states, quantity_data_emissions):
    state_labels = [str(x) for x in states]
    unique, counts = np.unique(quantity_data_states, return_counts=True)

    plt.bar(state_labels, counts)
    plt.title('State Distribution:')
    plt.xlabel('State')
    plt.ylabel('Quantity')
    plt.show()

    emission_labels = [str(x) for x in emissions]
    unique, counts = np.unique(quantity_data_emissions, return_counts=True)

    plt.bar(emission_labels,counts)
    plt.title('Emission Distribution:')
    plt.xlabel('Emission')
    plt.ylabel('Quantity')
    plt.show()

def matrix_plot(data, title, xlabs, ylabs):

    fig, ax = plt.subplots()
    plt.title(title)
    pos = ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(xlabs)), labels=xlabs)
    ax.set_yticks(np.arange(len(ylabs)), labels=ylabs)
    fig.colorbar(pos,ax=ax)
    plt.show()
