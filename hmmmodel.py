import math
import numpy as np
import random
import matplotlib.pyplot as plt
from hmmlearn import hmm

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
        resulting_emissions = []

        # Bepaal start state
        resulting_states = random.choices(self.states, weights=self.startprob)

        for i in range(n_steps-1):
            resulting_emissions.append(random.choices(self.emissions, weights=self.emissionprob[resulting_states[-1]])[0])
            resulting_states.append(random.choices(self.states, weights=self.transprob[resulting_states[-1]])[0])

        # Voeg laatste emissie toe
        resulting_emissions.append(random.choices(self.emissions, weights=self.emissionprob[resulting_states[-1]])[0])

        return resulting_states, resulting_emissions

    def visualize_results_barplot(self, quantity_data_states, quantity_data_emissions):
        state_labels = [str(x) for x in self.states]
        plt.bar(state_labels, [quantity_data_states.count(0), quantity_data_states.count(1), quantity_data_states.count(2)])
        plt.title('State Distribution:')
        plt.xlabel('State')
        plt.ylabel('Quantity')
        plt.show()

        emission_labels = [str(x) for x in self.emissions]
        plt.bar(emission_labels,
                [quantity_data_emissions.count(0), quantity_data_emissions.count(1), quantity_data_emissions.count(2),
                 quantity_data_emissions.count(3)])
        plt.title('Emission Distribution:')
        plt.xlabel('Emission')
        plt.ylabel('Quantity')
        plt.show()

    def visualize_results_matrix(self, resulting_states, resulting_emissions):

        # Determine observed transitions
        observed_transitions = np.zeros((3, 3))

        for i in range(1, len(resulting_states)-1):
            observed_transitions[resulting_states[i], resulting_states[i + 1]] += 1

        observed_transitions /= len(resulting_states) - 2

        observed_emissions = np.zeros((3, 4))

        if len(resulting_states) > 1:
            for i in range(len(observed_emissions)):
                observed_emissions[resulting_states[i], resulting_emissions[i]] += 1

        observed_emissions /= len(resulting_states) - 2

        return observed_transitions, observed_emissions

    def matrix_plot(self, data, title, xlabs, ylabs):

        # Observed Transprob
        fig, ax = plt.subplots()
        plt.title(title)
        pos = ax.imshow(data, cmap = "Blues")
        ax.set_xticks(np.arange(len(xlabs)), labels=xlabs)
        ax.set_yticks(np.arange(len(ylabs)), labels=ylabs)
        fig.colorbar(pos,ax=ax)
        plt.show()


    def score(self, state_seq, emission_seq):
        probability = self.startprob[state_seq[0]]
        log_probability = math.log(self.startprob[state_seq[0]])

        for i in range(0, len(state_seq)-1):
            probability_emission = self.emissionprob.get(state_seq[i])[emission_seq[i]]
            probability_table = self.transprob.get(state_seq[i-1])[state_seq[i]]
            probability *= probability_emission * probability_table

            if probability_emission > 0:
                log_probability += math.log(probability_emission) + math.log(probability_table)

            else:
                log_probability += math.log(probability_table)


        return probability, log_probability


def main():

    # Gelijke kans voor alle tafels als beginpunt
    startprob = np.array([1/3, 1/3, 1/3])

    # Transition probabilities
    transprob = {
        0:[1/6,1/2,1/3],
        1:[1/6,1/3,1/2],
        2:[2/3,1/6,1/6]
    }

    # Emission probabilities
    emissionprob = {
        0:[1/6, 1/12, 1/4, 1/2],
        1:[1/6, 1/6, 1/2, 1/6],
        2:[5/12, 1/2, 0, 1/12]
    }

    states = [0, 1, 2] # tafel 1 = 0, tafel 2 = 1, tafel 3 = 2
    emissions = [0, 1, 2, 3] # Blue = 0, Green = 1, Yellow = 2, Red = 3

    model = HiddenMarkovModel(startprob, transprob, emissionprob, states, emissions)
    resulting_states, resulting_emissions = model.sample(1200)

    print("-----resulting states-------")
    print(resulting_states)

    print("-----resulting emissions-------")
    print(resulting_emissions)


    model.visualize_results_barplot(resulting_states, resulting_emissions)


    #states_seq = resulting_states
    #emissions_seq = resulting_emissions

    states_seq =[1,1,1,1,1,2,0,2,1,2,0,0,2,0,0,1,2,0,2,0,1,0,1,2,0,1,2,0,1,1,1,1,1,1,2,0,2,1,2,0,0,2,0,0,1,2,0,2,0,1,0,1,2,0,1,2,0,1,1,1,1,1,1,2,0,2,1,2,0,0,2,0,0,1,2,0,2,0,1,0,1,2,0,1,2,0,1,1,1,1,1,1,2,0,2,1,2,0,0,2,0,0,1,2,0,2,0,1,0,1,2,0,1,2,0,1]
    emissions_seq = [2,3,3,0,1,2,0,2,3,2,2,0,2,0,1,2,2,0,3,0,1,3,1,2,0,0,2,0,1,2,3,3,0,1,2,0,2,3,2,2,0,2,0,1,2,2,0,3,0,1,3,1,2,0,0,2,0,1,2,3,3,0,1,2,0,2,3,2,2,0,2,0,1,2,2,0,3,0,1,3,1,2,0,0,2,0,1,2,3,3,0,1,2,0,2,3,2,2,0,2,0,1,2,2,0,3,0,1,3,1,2,0,0,2,0,1]

    probability, log_probability = model.score(states_seq, emissions_seq)
    print("-----probabilities-------")
    print("ln(p) = ", round(log_probability, 4))
    print("p = ", round(probability, 4))

    # Vergelijk geïmplementeerd hmm model met CategoricalHMM uit hmmlearn

    x = np.vstack((states_seq, emissions_seq))
    #x = [list(x) for x in zip(states_seq, emissions_seq)] # maak array van de sequences
    model1 = hmm.CategoricalHMM(n_components=3, n_features=4, init_params="")
    model1.transmat_ = np.array(list(transprob.values()))
    model1.emissionprob_ = np.array(list(emissionprob.values()))
    model1.startprob_ = startprob

    print("hmmlearn: ln(p)", round(model1.fit(x).score(x),4))

    ylabs = [str(x) for x in states]

    observed_transitions, observed_emissions = model.visualize_results_matrix(resulting_states, resulting_emissions)
    transprob_array = np.array(list(transprob.values()))
    model.matrix_plot(transprob_array, "Expected Transprob:",  xlabs=ylabs[::-1], ylabs = ylabs)
    model.matrix_plot(observed_transitions, "Observed Transprob:", xlabs=ylabs[::-1], ylabs = ylabs)

    xlabs = [str(x) for x in emissions]

    emissionprob_array = np.array(list(emissionprob.values()))

    model.matrix_plot(emissionprob_array, "Expected Emissionprob:", xlabs,  ylabs = ylabs)
    model.matrix_plot(observed_emissions, "Observed Emissionprob:", xlabs,  ylabs = ylabs)


if __name__ == "__main__":
    main()
