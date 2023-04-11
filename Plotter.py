import matplotlib.pyplot as plt

class Plotter:
    x = []
    y1 = []
    y2 = []

    def __init__(self, max_time):         
        plt.ion()
        plt.show()
        self.max_time = max_time

    def reset(self):
        self.x = []
        self.y1 = []
        self.y2 = []
        plt.cla()
    
    def plot(self, time, degree):
        plt.axis([0, self.max_time, -90, 90])
        self.x.append(time)
        self.y1.append(degree[0])
        self.y2.append(degree[1])
        plt.plot(self.x[len(self.x)-2:len(self.x)-1], self.y1[len(self.y1)-2:len(self.y1)-1], color='red', marker='o', linestyle='solid', markersize=2)
        plt.plot(self.x[len(self.x)-2:len(self.x)-1], self.y2[len(self.y2)-2:len(self.y2)-1], color='blue', marker='o', linestyle='solid', markersize=2)
        plt.draw()    
        plt.pause(0.001)