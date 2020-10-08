def feat_sel(model, feature_list):
    importances = list(model.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    feat_output = []
    feat_imp = []
    for feature_importance in feature_importances:
        if feature_importance[1] > 0:
            feat_output.append(feature_importance[0])
            feat_imp.append(feature_importance[1])

    return feat_output, feat_imp