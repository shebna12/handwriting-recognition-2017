import matplotlib.pyplot as plt


num = []
test_scores = [ 0.6834188 ,  0.70017094,  0.71333333,  0.71777778,  0.71777778,
        0.72888889,  0.72974359,  0.73452991,  0.73299145,  0.73760684,
        0.73726496,  0.73538462,  0.73726496,  0.73487179,  0.73264957,
        0.73726496,  0.73452991,  0.73675214,  0.73931624,  0.74102564,
        0.73709402,  0.73675214,  0.73863248,  0.73367521,  0.73863248,
        0.7391453 ,  0.73435897,  0.73897436,  0.73760684,  0.73367521,
        0.73401709,  0.73196581,  0.73863248,  0.73606838,  0.73863248,
        0.73623932,  0.73726496,  0.7374359 ,  0.73350427,  0.73709402,
        0.73965812]



a = [ 0.21623932,  0.34461538,  0.45384615,  0.54529915,  0.61179487,
        0.65384615,  0.68478632,  0.6965812 ,  0.70290598,  0.7182906 ,
        0.72034188,  0.72119658,  0.73384615,  0.73384615,  0.73487179,
        0.7357265 ,  0.73777778,  0.73401709,  0.73675214,  0.74461538,
        0.73811966,  0.74017094] 



b = [ 0.6834188 ,  0.70017094,  0.71333333,  0.71777778,  0.71777778,
        0.72888889,  0.72974359,  0.73452991,  0.73299145,  0.73760684,
        0.73726496,  0.73538462,  0.73726496,  0.73487179,  0.73264957,
        0.73726496,  0.73452991,  0.73675214,  0.73931624,  0.74102564,
        0.73709402,  0.73675214,  0.73863248,  0.73367521,  0.73863248,
        0.7391453 ,  0.73435897,  0.73897436,  0.73760684,  0.73367521,
        0.73401709,  0.73196581,  0.73863248,  0.73606838,  0.73863248,
        0.73623932,  0.73726496,  0.7374359 ,  0.73350427,  0.73709402,
        0.73965812 ]

c= [ 0.73880342,  0.73555556,  0.73846154,  0.73777778,  0.73794872,
        0.73350427,  0.73470085,  0.73709402,  0.73470085,  0.73863248,
        0.73452991,  0.72940171,  0.73555556,  0.73692308,  0.73863248,
        0.73863248,  0.73606838,  0.74444444,  0.73948718,  0.73487179,
        0.74119658,  0.73401709,  0.73931624,  0.74102564,  0.73948718,
        0.74188034,  0.7357265 ,  0.73777778,  0.73982906,  0.73675214,
        0.73709402,  0.73811966,  0.73333333,  0.7408547 ,  0.73384615,
        0.74051282,  0.73760684,  0.73247863,  0.73487179,  0.73299145,
        0.7408547 ,  0.74136752,  0.73487179,  0.74068376,  0.7391453 ,
        0.74153846,  0.73863248,  0.73709402,  0.74017094,  0.74632479 ]

print(max(a))
print(max(b))
print(max(c))
# print(len(test_scores))
import sys
sys.exit(0)
for x in range(9,50):
	num.append(x)
print(num)

plt.plot(num,test_scores,linewidth=2.0)
plt.show()
