# CT2Xray
這是ct2xray專案 主要用於miccai資料集 為了產生奇美骨裂專案的擴增資料集

```py
#測試 


def scale_range(array_2d, min_v, max_v, threshold_low=0.0, threshold_high=1.0):#compress image form p view to ont view
    np_min = np.min(array_2d)
    np_max = np.max(array_2d)
    for h in range(len(array_2d)):
        for w in range(len(array_2d[0])):
            if array_2d[h][w] <= int(threshold_low * (np_max - np_min)):
                array_2d[h][w] = 0
                continue
            if array_2d[h][w] >= int(threshold_high * (np_max - np_min)):
                array_2d[h][w] = 255
                continue
            array_2d[h][w] -= np_min
            array_2d[h][w] /= np_max / (max_v - min_v)
            array_2d[h][w] += min_v

        print("\rSuccessful Max: {}, Min {}.".format(array_2d.max(), array_2d.min()),
              end='')
    return array_2d
```
