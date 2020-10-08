def feat_agg(feat_output, feat_imp, dict_feat):
    for i, feat in enumerate(feat_output):
        if feat in dict_feat.keys():
            dict_feat[feat].append(feat_imp[i])
        else:
            dict_feat[feat] = [feat_imp[i]]

    return dict_feat