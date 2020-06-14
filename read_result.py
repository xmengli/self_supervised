import numpy as np
import argparse



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('result', metavar='DIR',
#                     help='path of result')
# parser.add_argument('epoch', type=int, default=2000)
args = parser.parse_args()

# def read_result():
#
#     result = []
#     for i in list(range(0, 5)):
#         data = np.genfromtxt(args.result + str(i) + "/result.txt", usecols=1, dtype=float)
#         num = int(args.epoch / 200)
#
#         selected = data[num*6:num*6+6]
#         selected = selected[1:]
#         result.append(selected)
#
#         # tmp = data[(num-1)*6:(num-1)*6+6]
#         # result.append(tmp[1:])
#         # tmp = data[(num - 2) * 6:(num-2) * 6 + 6]
#         # result.append(tmp[1:])
#
#     result = np.stack(result)
#     print (result)
#     mean, std = np.mean(result, axis=0), np.std(result, axis=0)
#     print ("mean ", np.around(mean*100, decimals=2))
#     print ("std ", np.around(std*100, decimals=2))

def read_txtfile():
    data = np.genfromtxt("savedmodels/result.txt", usecols=1, dtype=float)
    results = np.reshape(data, (5,5))
    a = np.mean(results,axis=0)
    print ("5-fold result: ")
    print ("AUC", np.around(a[0]*100, decimals=2))
    print ("acc", np.around(a[1]*100, decimals=2))
    print ("precision", np.around(a[2]*100, decimals=2))
    print ("recall", np.around(a[3]*100, decimals=2))
    print ("f1score", np.around(a[4]*100, decimals=2))


if __name__ == '__main__':

    read_txtfile()

