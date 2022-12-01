import numpy as np
import matplotlib.pyplot as plt

class Plot_Confusion_Matrix:
    def __init__(self, class_name):
        self.class_name = class_name
        self.class_num = len(class_name)
        self.CM = np.zeros((self.class_num, self.class_num), dtype='float32')

    def append(self, gt_idx, pred_idx):
        self.CM[gt_idx, pred_idx] += 1

    def plot(self):
        sum_in_line = self.CM.sum(axis=1)
        for i in range(self.class_num):
            self.CM[i] = (self.CM[i] / sum_in_line[i])
        self.CM = np.around(self.CM, 2)
        self.CM[np.isnan(self.CM)] = 0
        plt.imshow(self.matrix, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        plt.xticks(range(self.class_num), self.class_name, rotation=45)
        plt.yticks(range(self.class_num), self.class_name)

        for x in range(self.class_num):
            for y in range(self.class_num):
                v = float(format('%.2f' % self.CM[x, y]))
                plt.text(x, y, v, verticalalignment='center', horizontalalignment='center')

        plt.tight_layout()
        plt.colorbar()
        plt.savefig('./CM.pnt', bbox_inches='tight')
        plt.show()