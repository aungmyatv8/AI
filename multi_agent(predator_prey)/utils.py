import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(x, scores):
    fig = plt.figure()
    ax = fig.add_subplot(121, label="1")
    ax2 = fig.add_subplot(122, label="2")   
    x = [i+1 for i in range(x)] 

    ax.plot(x, scores["agent0"], color='r', linestyle='--', label="Adversary Agent")
    ax.plot(x, scores["agent3"], color='g', label="Good Agent")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Scores")
    ax.set_title("Good Agent Vs Adversary Agent")
    ax.legend()

    N = len(scores['agent0'])
    running_avg = np.empty(N)
    # calculate average score
    for t in range(N):
        running_avg[t] = np.mean(scores["agent0"][t] * 3 + scores["agent3"][t])

    ax2.plot(x, running_avg, color="C")
    ax2.set_xlabel("Number of games")
    ax2.set_ylabel('Score', color="C")
    ax2.set_title("Average Score of All agents")

    plt.savefig("plot.png")
    # plt.show()