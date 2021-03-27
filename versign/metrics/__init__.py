from matplotlib import pyplot as plt
from scipy import optimize


def accuracy_scores(predictions, actual):
    fars = []
    frrs = []
    for thresh, predicted in predictions.items():
        far, frr = accuracy_score(predicted, actual)
        fars.append(far)
        frrs.append(frr)

    return fars, frrs


def accuracy_score(predicted, actual):
    assert isinstance(predicted, list) and isinstance(actual, list)
    assert len(predicted) == len(actual)

    genuine = 0
    forged = 0
    false_pos = 0
    false_neg = 0
    for y_pred, y_actual in zip(predicted, actual):
        if y_actual == 1:
            genuine += 1
            if y_pred == -1:
                false_neg += 1
        else:
            forged += 1
            if y_pred == 1:
                false_pos += 1

    far = 0.0
    if false_pos != 0:
        far = false_pos / forged

    frr = 0.0
    if false_neg != 0:
        frr = false_neg / genuine

    return far, frr


def calc_equal_error(FARs, FRRs, sensitivity):
    def parabola(x, a, b, c):
        return a*x**2 + b*x + c

    FARs *= 100
    FRRs *= 100

    plt.scatter(sensitivity, FARs, s=1)
    f1, params_cov = optimize.curve_fit(parabola, sensitivity, FARs)
    plt.plot(sensitivity, parabola(sensitivity, f1[0], f1[1], f1[2]), label='False Acceptance')

    plt.scatter(sensitivity, FRRs, s=1)
    f2, params_cov = optimize.curve_fit(parabola, sensitivity, FRRs)
    plt.plot(sensitivity, parabola(sensitivity, f2[0], f2[1], f2[2]), label='False Rejection')

    intersect = optimize.fsolve(lambda x : parabola(x, f1[0], f1[1], f1[2]) - parabola(x, f2[0], f2[1], f2[2]), sensitivity[:1])
    eer = parabola(intersect, f2[0], f2[1], f2[2])[0]
    plt.scatter(intersect, eer, label=f'ERR={eer:.2f}%')

    plt.legend(loc='best')
    plt.xlabel('Threshold')
    plt.ylabel('Error (%)')
    return eer, plt
