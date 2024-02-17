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
                      df: pd.DataFrame):
    # drop any nulls, then extract x and y
    df = df[[num_feat, group_feat]]
    df = df.dropna(subset=[num_feat, group_feat])
    df_x = df[[num_feat]]
    df_y = df[group_feat].map(map_target)

    # data preprocessors
    scaler = StandardScaler()
    ct = make_column_transformer(
        (scaler, [num_feat]))

    # train the logreg
    logreg = LogisticRegression(solver='liblinear')
    pipe1 = make_pipeline(ct, logreg)
    print(f"[INFO] training the logreg")
    pipe1.fit(df_x, df_y)

    # look at coeffs via statsmodels
    print(f"[INFO] getting the summary via statsmodels")
    pipe2 = sm.Logit(df_y, ct.transform(df_x)).fit()
    print(pipe2.summary())

    print(f"[INFO] getting predictions and plotting")
    df_x_probs = pipe1.predict_proba(df_x)[:, 1]
    logreg_res = pd.DataFrame({"prob": list(df_x_probs),
                               num_feat: df_x[num_feat]})

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
    return logreg_res
