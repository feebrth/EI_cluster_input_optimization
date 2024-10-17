import pickle
import matplotlib.pyplot as plt




def get_exp_data(condition):
    with open('Result_dict.pkl', 'rb') as f:
        experimentally_observed = pickle.load(f)

    time = experimentally_observed['time_rate']
    firing_rate_1 = experimentally_observed[condition]['rate']

    return time, firing_rate_1



if __name__ == "__main__":

    time, firing = get_exp_data(1)

    print(get_exp_data(1))

    plt.plot(get_exp_data(1)[1])
    plt.plot(get_exp_data(2)[1])
    plt.plot(get_exp_data(3)[1])
    plt.show()
    # time = experimentally_observed['time_rate']
    # firing_rate_1 = experimentally_observed[1]['rate']
    # firing_rate_2 = experimentally_observed[2]['rate']
    # firing_rate_3 = experimentally_observed[3]['rate']
    #
    # plt.plot(time, firing_rate_1, label= 'Firing Rate 1', color= 'blue')
    # plt.plot(time, firing_rate_2, label= 'Firing Rate 2', color= 'red')
    # plt.plot(time, firing_rate_3, label= 'Firing Rate3', color= 'green')
    #
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Firing Rate (spikes/s)')
    #
    # plt.legend()
    #
    # plt.show()