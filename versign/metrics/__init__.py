def accuracy_score(results, ground_truth):
    right = 0
    wrong = 0
    false_pos = 0
    false_neg = 0
    accuracy = 0
    details = dict()
    for i, y_test in enumerate(results):
        y_true = ground_truth[i]

        # Save stats
        pred = y_true == y_test
        mistakes = [i for i, x in enumerate(pred) if not x]

        fpos = 0
        fneg = 0
        for mistake in mistakes:
            if y_true[mistake] == 1 and y_test[mistake] == -1:
                fneg += 1
            else:
                fpos += 1

        details[i] = {
            'right': len(pred[pred == True]),
            'wrong': len(pred[pred == False]),
            'false-pos': fpos,
            'false-neg': fneg,
            'accuracy': len(pred[pred == True]) / len(y_test) * 100
        }

        right += details[i]['right']
        wrong += details[i]['wrong']
        false_pos += details[i]['false-pos']
        false_neg += details[i]['false-neg']
        accuracy += details[i]['accuracy']

    accuracy /= len(details.keys())
    summary = {
        '# of Users': len(details.keys()),
        'Test Samples': right + wrong,
        'Average Accuracy': round(accuracy, 2),
        'False Positives': round(false_pos / (right + wrong) * 100, 2),
        'False Negatives': round(false_neg / (right + wrong) * 100, 2)
    }
    return summary, details
