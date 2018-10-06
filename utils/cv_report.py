from matplotlib import pyplot as plt


def cv_plot(cv_scores, scale, param,
            min_k=None, min_cv=None,
            best_k=None, best_cv=None,
            word=True,
            figsize=(6, 4),
            y_lim=None):

    """ plot MSE with parameter, the smallest rc score and the best parameter picked accordinly"""
    # plot mse with 1 std error bar
    plt.figure(figsize=figsize)
    plt.errorbar(x=scale, y=[s[0] for s in cv_scores],
                 yerr=[s[1] for s in cv_scores], xerr=None, fmt='',
                 ecolor=None, elinewidth=None, capsize=None,
                 barsabove=False, lolims=False, uplims=False,
                 xlolims=False, xuplims=False)

    # Annotate the smallest and the better k (simpler model)
    if min_k and min_cv and best_k and best_cv:
        for x, y, label in [[min_k, min_cv, "smallest cv score"], [best_k, best_cv, "better(simpler) " + param]]:
            if word: print("min:    {}, with score {}".format(x,
                                                              y) if label == "smallest cv score" else "better: {}, with score {}".format(
                x, y))
            plt.plot([x], [y], 'ro')
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])

    plt.xlabel(param)
    plt.ylabel("Cross-validation score (training)")
    plt.show()


def cv_report(cv_scores, scale, param="k",
              display=True, plot=True, name="", forward=True):
    """
    give a cv report for sv score results of 10-fold cross validation from training some model:
    analyze the data and select the best model,
    e,g, k in K-NN, lambda in Lasso and Ridge.

    Args:
       cv_scores: 2D array of cv scores (mean, std)
       scale: x scale function
       param: string, such as "k" or "lambda"
       display: boolean, if print working process
       plot: boolean, if plot cv result
       forward: true if maximize parameter as simpler model
    Return:
       best_k: parameter for a simpler model
       best_cv: cv score for the best k
       min_k: parameter that could predict the best score
       min_cv: cv score for the min k
    """

    # header
    if display: print(
        "\n===========================================================\n                   CV Score Report ({}):\n===========================================================".format(
            name))

    # find k with the smallest cv score
    best_k, min_k, min_cv, std = -1, -1, float("inf"), 0
    for i, k in enumerate(scale):
        m, s = cv_scores[i]
        if abs(m) < abs(min_cv):
            min_k, min_cv, std = k, m, s

    # report smallest k
    if display: print(
        "{} = {} has the smallest CV score of:\n mean                std\n{},  {}".format(param, min_k, min_cv, std))
    if display: print(
        "\nAmong all " + param + " reporting a score within 1 SE of the \nsmallest score({}, {}),\nwe choose the simplest model as a better model. \n".format(
            min_cv - std, min_cv + std))

    # Filter effective k within +/- 1 std error of smallest k
    if display: print("all " + param + " that qualifies")
    if display: print(param + "    cv score   std")

    display_table = []
    best_k, best_cv = min_k, min_cv

    for i, k in enumerate(scale):
        cv, st = cv_scores[i]
        lower, higher = max(abs(min_cv) - std, 0), abs(min_cv) + std
        if lower <= abs(cv) <= higher:
            display_table.append(
                "{}: {}{} | {} |\n".format(str(k)[:4], '  ' if k < 10 else ' ', str(cv)[:17], str(st)[:15]))

            # principled rule: find simplest model possible, larger k
            if (forward and k > best_k) or (not forward and k < best_k): best_k, best_cv = k, cv

    if best_k < 0: display_table.append("none none    none")

    # display k table
    if display:
        if len(display_table) > 20:
            print("".join(display_table[:6]) + "          .............................  \n" * 2 + "\n" + "".join(
                display_table[-5:]))
        else:
            print("".join(display_table))

    if display: print(
        "\nWe could risk a simpler model, i.e. pick better " + param + " = {} with cv score of {}\n------------------------------------------------------------".format(
            best_k, best_cv))
    # plot cv against k
    if plot:
        print("Below plots the CV error against parameter {}, and mark\n".format(param))
        cv_plot(cv_scores, scale, param, min_k, min_cv, best_k, best_cv)

    return best_k, best_cv, min_k, min_cv