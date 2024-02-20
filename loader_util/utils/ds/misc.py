import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %% ##################################################################
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


# %% ##################################################################
def get_logistic_plot(num_feat: str,
                      group_feat: str,
                      map_target: dict,
                      df: pd.DataFrame,
                      scale=True,
                      solver='liblinear'):
    # drop any nulls, then extract x and y
    df = df[[num_feat, group_feat]]
    df = df.dropna(subset=[num_feat, group_feat])
    df_x = df[[num_feat]]
    df_y = df[group_feat].map(map_target)

    logreg = LogisticRegression(solver=solver, )

    # data preprocessors
    if scale:
        scaler = StandardScaler()
        ct = make_column_transformer(
            (scaler, [num_feat]))
        pipe1 = make_pipeline(ct, logreg)

    else:
        pipe1 = make_pipeline(logreg)

    print(f"[INFO] training the logreg")
    pipe1.fit(df_x, df_y)

    if scale:
        pipe2 = sm.Logit(df_y, ct.transform(df_x))
    else:
        pipe2 = sm.Logit(df_y, df_x)
    pipe2 = pipe2.fit()

    # getting sklearn coef
    print(f"[INFO] getting sklearn params")
    param_df = pd.DataFrame({
        'coef_': pipe1.named_steps['logisticregression'].coef_[0],
        'intercept_': pipe1.named_steps['logisticregression'].intercept_[0]
    })

    # look at coeffs via statsmodels
    print(f"[INFO] getting the summary via statsmodels")
    print(pipe2.summary())
    print("=" * 50)

    print(f"[INFO] getting predictions and plotting")
    df_x_probs = pipe1.predict_proba(df_x)[:, 1]
    logreg_res = pd.DataFrame({"prob": list(df_x_probs),
                               num_feat: df_x[num_feat]})
    print("=" * 50)

    # do the logistic plot
    sns.set(font_scale=1.5)
    sns.set_style("darkgrid", rc={"grid.color": ".20",
                                  "grid.linestyle": ":"})
    palette = sns.color_palette("tab10")
    sns.set_palette(palette=palette)
    nrows = 1
    ncols = 1
    fig_height = nrows * 4
    fig_width = ncols * 6

    f, ax = plt.subplots(nrows=nrows,
                         ncols=ncols,
                         figsize=(fig_width, fig_height))
    sns.scatterplot(data=logreg_res,
                    x=num_feat,
                    y="prob",
                    ax=ax,
                    linewidth=2,
                    color="blue")
    ax.set(ylim=(0, 1),
           xlim=(logreg_res[num_feat].min(), None))
    return param_df, logreg_res
