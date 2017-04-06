

```python
import numpy as np
from NeuralNetwork import NeuralNetwork
from load_mnist import MNIST_Loader
from error_validation import *
print "loaded"
```

    loaded


$ x_{i} $


```python
a = MNIST_Loader()
X_train, y_train = a.load_mnist('./data')
X_test, y_test = a.load_mnist('./data', 't10k')
print X_train.shape
print X_test.shape
print X_train[0]
```

    (60000, 784)
    (10000, 784)
    [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255
     247 127   0   0   0   0   0   0   0   0   0   0   0   0  30  36  94 154
     170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0   0   0
       0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251  93  82
      82  56  39   0   0   0   0   0   0   0   0   0   0   0   0  18 219 253
     253 253 253 253 198 182 247 241   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0  14   1 154 253  90   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0  11 190 253  70   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  35 241
     225 160 108   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0  81 240 253 253 119  25   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253
     253 207   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253
     253 201  78   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0  23  66 213 253 253 253 253 198  81   2   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0  18 171 219 253 253 253 253 195
      80   9   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
      55 172 226 253 253 253 253 244 133  11   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0 136 253 253 253 212 135 132  16
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
       0   0   0   0   0   0   0   0   0   0]



```python

```


```python
print y_train.shape
print y_test.shape
```

    (60000,)
    (10000,)



```python

```


```python
nn = NeuralNetwork(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=100,
                  l2=0.1,
                  epochs=1000,
                  learning_rate=0.001,
                  momentum_const=0.5,
                  decay_rate=0.00001,
                  dropout=True,
                  minibatch_size=500,
                 )

```


```python

```


```python
nn.fit(X_train, y_train, print_progress=True)

```

    Epoch: 1
    Loss: 11.4863186906
    Training Accuracy: 72.3316666667
    Epoch: 2
    Loss: 9.96101675805
    Training Accuracy: 77.6033333333
    Epoch: 3
    Loss: 8.79242232026
    Training Accuracy: 81.3816666667
    Epoch: 4
    Loss: 7.67417013288
    Training Accuracy: 81.8216666667
    Epoch: 5
    Loss: 6.86434756793
    Training Accuracy: 82.4183333333
    Epoch: 6
    Loss: 6.21540978285
    Training Accuracy: 84.2866666667
    Epoch: 7
    Loss: 5.48939117542
    Training Accuracy: 83.8516666667
    Epoch: 8
    Loss: 5.02466623227
    Training Accuracy: 84.2883333333
    Epoch: 9
    Loss: 4.38899386563
    Training Accuracy: 85.065
    Epoch: 10
    Loss: 4.1225288191
    Training Accuracy: 85.8333333333
    Epoch: 11
    Loss: 3.60661741608
    Training Accuracy: 85.6433333333
    Epoch: 12
    Loss: 3.32130372311
    Training Accuracy: 85.415
    Epoch: 13
    Loss: 3.12704554356
    Training Accuracy: 86.6683333333
    Epoch: 14
    Loss: 2.95022814615
    Training Accuracy: 85.2333333333
    Epoch: 15
    Loss: 2.78257690437
    Training Accuracy: 86.695
    Epoch: 16
    Loss: 2.59735603243
    Training Accuracy: 87.62
    Epoch: 17
    Loss: 2.57316813042
    Training Accuracy: 86.8433333333
    Epoch: 18
    Loss: 2.35519925188
    Training Accuracy: 87.15
    Epoch: 19
    Loss: 2.30072864861
    Training Accuracy: 86.3716666667
    Epoch: 20
    Loss: 2.09264247987
    Training Accuracy: 86.6866666667
    Epoch: 21
    Loss: 2.02089140513
    Training Accuracy: 87.43
    Epoch: 22
    Loss: 1.87384529019
    Training Accuracy: 87.545
    Epoch: 23
    Loss: 2.01161874031
    Training Accuracy: 87.5266666667
    Epoch: 24
    Loss: 1.93545063244
    Training Accuracy: 86.9383333333
    Epoch: 25
    Loss: 1.97518485957
    Training Accuracy: 87.745
    Epoch: 26
    Loss: 1.84365427507
    Training Accuracy: 87.9216666667
    Epoch: 27
    Loss: 1.76063144692
    Training Accuracy: 86.6066666667
    Epoch: 28
    Loss: 1.92910471683
    Training Accuracy: 87.2566666667
    Epoch: 29
    Loss: 1.83810432939
    Training Accuracy: 86.465
    Epoch: 30
    Loss: 1.94834966733
    Training Accuracy: 87.0866666667
    Epoch: 31
    Loss: 1.79249844874
    Training Accuracy: 86.59
    Epoch: 32
    Loss: 1.94786271409
    Training Accuracy: 87.1583333333
    Epoch: 33
    Loss: 1.84410236577
    Training Accuracy: 87.4983333333
    Epoch: 34
    Loss: 1.78089678144
    Training Accuracy: 86.7066666667
    Epoch: 35
    Loss: 1.90232460126
    Training Accuracy: 87.355
    Epoch: 36
    Loss: 1.66279323437
    Training Accuracy: 87.3533333333
    Epoch: 37
    Loss: 1.8495530377
    Training Accuracy: 87.645
    Epoch: 38
    Loss: 1.83763016732
    Training Accuracy: 87.75
    Epoch: 39
    Loss: 1.73801953701
    Training Accuracy: 87.835
    Epoch: 40
    Loss: 1.76080049355
    Training Accuracy: 87.8083333333
    Epoch: 41
    Loss: 1.72005611867
    Training Accuracy: 87.7133333333
    Epoch: 42
    Loss: 1.64463986957
    Training Accuracy: 87.985
    Epoch: 43
    Loss: 1.78245046971
    Training Accuracy: 88.0116666667
    Epoch: 44
    Loss: 1.68990969433
    Training Accuracy: 88.2966666667
    Epoch: 45
    Loss: 1.65815029372
    Training Accuracy: 88.3233333333
    Epoch: 46
    Loss: 1.76690587667
    Training Accuracy: 87.4666666667
    Epoch: 47
    Loss: 1.65396960715
    Training Accuracy: 87.97
    Epoch: 48
    Loss: 1.67016989765
    Training Accuracy: 87.875
    Epoch: 49
    Loss: 1.63500896345
    Training Accuracy: 87.9
    Epoch: 50
    Loss: 1.74454094906
    Training Accuracy: 87.7616666667
    Epoch: 51
    Loss: 1.73249982236
    Training Accuracy: 87.51
    Epoch: 52
    Loss: 1.69720026093
    Training Accuracy: 87.78
    Epoch: 53
    Loss: 1.67204144533
    Training Accuracy: 87.665
    Epoch: 54
    Loss: 1.7915602708
    Training Accuracy: 87.8666666667
    Epoch: 55
    Loss: 1.61625530496
    Training Accuracy: 87.3866666667
    Epoch: 56
    Loss: 1.75566330926
    Training Accuracy: 87.275
    Epoch: 57
    Loss: 1.7323454866
    Training Accuracy: 88.135
    Epoch: 58
    Loss: 1.68533418371
    Training Accuracy: 87.4633333333
    Epoch: 59
    Loss: 1.57664432815
    Training Accuracy: 87.4666666667
    Epoch: 60
    Loss: 1.58079258499
    Training Accuracy: 87.9816666667
    Epoch: 61
    Loss: 1.77328064955
    Training Accuracy: 87.7716666667
    Epoch: 62
    Loss: 1.77379542635
    Training Accuracy: 87.4083333333
    Epoch: 63
    Loss: 1.78901506086
    Training Accuracy: 87.72
    Epoch: 64
    Loss: 1.76184927807
    Training Accuracy: 87.8633333333
    Epoch: 65
    Loss: 1.73539331149
    Training Accuracy: 87.1466666667
    Epoch: 66
    Loss: 1.73247109933
    Training Accuracy: 87.7783333333
    Epoch: 67
    Loss: 1.72600392915
    Training Accuracy: 87.73
    Epoch: 68
    Loss: 1.68576605368
    Training Accuracy: 88.0966666667
    Epoch: 69
    Loss: 1.71021887363
    Training Accuracy: 87.475
    Epoch: 70
    Loss: 1.80132104645
    Training Accuracy: 87.3066666667
    Epoch: 71
    Loss: 1.58748218452
    Training Accuracy: 87.33
    Epoch: 72
    Loss: 1.71121343781
    Training Accuracy: 87.7733333333
    Epoch: 73
    Loss: 1.73941132835
    Training Accuracy: 87.545
    Epoch: 74
    Loss: 1.71702767111
    Training Accuracy: 86.9433333333
    Epoch: 75
    Loss: 1.76708665137
    Training Accuracy: 87.195
    Epoch: 76
    Loss: 1.75792549956
    Training Accuracy: 87.025
    Epoch: 77
    Loss: 1.73419107272
    Training Accuracy: 86.4333333333
    Epoch: 78
    Loss: 1.73454413932
    Training Accuracy: 87.3416666667
    Epoch: 79
    Loss: 1.62456091088
    Training Accuracy: 87.3566666667
    Epoch: 80
    Loss: 1.60979252225
    Training Accuracy: 87.5433333333
    Epoch: 81
    Loss: 1.66751965039
    Training Accuracy: 87.3966666667
    Epoch: 82
    Loss: 1.62250768748
    Training Accuracy: 87.755
    Epoch: 83
    Loss: 1.64304659123
    Training Accuracy: 87.8833333333
    Epoch: 84
    Loss: 1.75796352961
    Training Accuracy: 87.7316666667
    Epoch: 85
    Loss: 1.74790936435
    Training Accuracy: 87.75
    Epoch: 86
    Loss: 1.71049916489
    Training Accuracy: 87.96
    Epoch: 87
    Loss: 1.68268786859
    Training Accuracy: 87.8516666667
    Epoch: 88
    Loss: 1.65806751693
    Training Accuracy: 87.5216666667
    Epoch: 89
    Loss: 1.62831148205
    Training Accuracy: 87.5966666667
    Epoch: 90
    Loss: 1.70275808421
    Training Accuracy: 88.0683333333
    Epoch: 91
    Loss: 1.69189146428
    Training Accuracy: 87.165
    Epoch: 92
    Loss: 1.73934576475
    Training Accuracy: 87.435
    Epoch: 93
    Loss: 1.7055709905
    Training Accuracy: 86.8566666667
    Epoch: 94
    Loss: 1.64538390203
    Training Accuracy: 87.4316666667
    Epoch: 95
    Loss: 1.6471513812
    Training Accuracy: 87.5866666667
    Epoch: 96
    Loss: 1.64755251804
    Training Accuracy: 87.5633333333
    Epoch: 97
    Loss: 1.75904170597
    Training Accuracy: 87.975
    Epoch: 98
    Loss: 1.65445851563
    Training Accuracy: 87.76
    Epoch: 99
    Loss: 1.68180590857
    Training Accuracy: 87.685
    Epoch: 100
    Loss: 1.59884659944
    Training Accuracy: 88.065
    Epoch: 101
    Loss: 1.80940465397
    Training Accuracy: 88.3283333333
    Epoch: 102
    Loss: 1.71799530627
    Training Accuracy: 87.765
    Epoch: 103
    Loss: 1.62966672597
    Training Accuracy: 88.1233333333
    Epoch: 104
    Loss: 1.79134246078
    Training Accuracy: 87.845
    Epoch: 105
    Loss: 1.68402047463
    Training Accuracy: 88.4116666667
    Epoch: 106
    Loss: 1.6719334414
    Training Accuracy: 87.945
    Epoch: 107
    Loss: 1.65044263275
    Training Accuracy: 87.8966666667
    Epoch: 108
    Loss: 1.68510469234
    Training Accuracy: 87.8666666667
    Epoch: 109
    Loss: 1.74278915766
    Training Accuracy: 87.38
    Epoch: 110
    Loss: 1.57187753705
    Training Accuracy: 88.365
    Epoch: 111
    Loss: 1.56701933243
    Training Accuracy: 88.2083333333
    Epoch: 112
    Loss: 1.71344277534
    Training Accuracy: 87.6683333333
    Epoch: 113
    Loss: 1.71536400066
    Training Accuracy: 87.5233333333
    Epoch: 114
    Loss: 1.72853814863
    Training Accuracy: 87.74
    Epoch: 115
    Loss: 1.67555124781
    Training Accuracy: 88.29
    Epoch: 116
    Loss: 1.66977894397
    Training Accuracy: 87.77
    Epoch: 117
    Loss: 1.59997567052
    Training Accuracy: 88.13
    Epoch: 118
    Loss: 1.56667706585
    Training Accuracy: 88.3216666667
    Epoch: 119
    Loss: 1.69199487892
    Training Accuracy: 87.675
    Epoch: 120
    Loss: 1.64918814992
    Training Accuracy: 87.82
    Epoch: 121
    Loss: 1.71272130487
    Training Accuracy: 87.3766666667
    Epoch: 122
    Loss: 1.6827313301
    Training Accuracy: 87.9466666667
    Epoch: 123
    Loss: 1.64034417477
    Training Accuracy: 88.0366666667
    Epoch: 124
    Loss: 1.62346329628
    Training Accuracy: 88.4516666667
    Epoch: 125
    Loss: 1.57549115893
    Training Accuracy: 88.0033333333
    Epoch: 126
    Loss: 1.8167715978
    Training Accuracy: 88.0616666667
    Epoch: 127
    Loss: 1.64124277657
    Training Accuracy: 88.4733333333
    Epoch: 128
    Loss: 1.67556210752
    Training Accuracy: 88.305
    Epoch: 129
    Loss: 1.64397357127
    Training Accuracy: 87.805
    Epoch: 130
    Loss: 1.63784787799
    Training Accuracy: 88.4366666667
    Epoch: 131
    Loss: 1.57709003579
    Training Accuracy: 88.4616666667
    Epoch: 132
    Loss: 1.62461939449
    Training Accuracy: 87.815
    Epoch: 133
    Loss: 1.59906653206
    Training Accuracy: 88.2816666667
    Epoch: 134
    Loss: 1.67072125451
    Training Accuracy: 88.47
    Epoch: 135
    Loss: 1.44600718353
    Training Accuracy: 88.0666666667
    Epoch: 136
    Loss: 1.69540839261
    Training Accuracy: 88.5366666667
    Epoch: 137
    Loss: 1.6633212559
    Training Accuracy: 88.3383333333
    Epoch: 138
    Loss: 1.6385122823
    Training Accuracy: 88.045
    Epoch: 139
    Loss: 1.62373064308
    Training Accuracy: 87.9883333333
    Epoch: 140
    Loss: 1.7188473401
    Training Accuracy: 87.835
    Epoch: 141
    Loss: 1.59578334279
    Training Accuracy: 87.9383333333
    Epoch: 142
    Loss: 1.6205834002
    Training Accuracy: 87.7633333333
    Epoch: 143
    Loss: 1.53972807362
    Training Accuracy: 88.45
    Epoch: 144
    Loss: 1.61791361723
    Training Accuracy: 88.2283333333
    Epoch: 145
    Loss: 1.59903898166
    Training Accuracy: 87.5416666667
    Epoch: 146
    Loss: 1.70635194831
    Training Accuracy: 88.8116666667
    Epoch: 147
    Loss: 1.51059291801
    Training Accuracy: 87.4933333333
    Epoch: 148
    Loss: 1.54842424316
    Training Accuracy: 87.8733333333
    Epoch: 149
    Loss: 1.57416683316
    Training Accuracy: 88.2983333333
    Epoch: 150
    Loss: 1.63882175431
    Training Accuracy: 87.5683333333
    Epoch: 151
    Loss: 1.63482910931
    Training Accuracy: 87.59
    Epoch: 152
    Loss: 1.57304264772
    Training Accuracy: 88.3866666667
    Epoch: 153
    Loss: 1.57125781903
    Training Accuracy: 88.1333333333
    Epoch: 154
    Loss: 1.61383224199
    Training Accuracy: 88.4033333333
    Epoch: 155
    Loss: 1.5379196737
    Training Accuracy: 88.7283333333
    Epoch: 156
    Loss: 1.5760986762
    Training Accuracy: 87.8683333333
    Epoch: 157
    Loss: 1.73222045467
    Training Accuracy: 88.0466666667
    Epoch: 158
    Loss: 1.69593370109
    Training Accuracy: 88.0416666667
    Epoch: 159
    Loss: 1.64485595642
    Training Accuracy: 88.4033333333
    Epoch: 160
    Loss: 1.47459065443
    Training Accuracy: 88.1766666667
    Epoch: 161
    Loss: 1.68215067577
    Training Accuracy: 88.1383333333
    Epoch: 162
    Loss: 1.62699486243
    Training Accuracy: 88.075
    Epoch: 163
    Loss: 1.58008996397
    Training Accuracy: 87.9883333333
    Epoch: 164
    Loss: 1.53023831517
    Training Accuracy: 89.215
    Epoch: 165
    Loss: 1.49405977069
    Training Accuracy: 88.3616666667
    Epoch: 166
    Loss: 1.54378637133
    Training Accuracy: 87.7583333333
    Epoch: 167
    Loss: 1.52368425456
    Training Accuracy: 88.31
    Epoch: 168
    Loss: 1.6324398057
    Training Accuracy: 88.5533333333
    Epoch: 169
    Loss: 1.5021500182
    Training Accuracy: 88.2566666667
    Epoch: 170
    Loss: 1.49524623782
    Training Accuracy: 88.2
    Epoch: 171
    Loss: 1.73810625162
    Training Accuracy: 87.8733333333
    Epoch: 172
    Loss: 1.62438087663
    Training Accuracy: 88.03
    Epoch: 173
    Loss: 1.49051825743
    Training Accuracy: 88.2283333333
    Epoch: 174
    Loss: 1.55297874393
    Training Accuracy: 88.2366666667
    Epoch: 175
    Loss: 1.50974215173
    Training Accuracy: 88.5933333333
    Epoch: 176
    Loss: 1.650593541
    Training Accuracy: 88.175
    Epoch: 177
    Loss: 1.58546716473
    Training Accuracy: 88.0916666667
    Epoch: 178
    Loss: 1.55737681381
    Training Accuracy: 88.9083333333
    Epoch: 179
    Loss: 1.51195086901
    Training Accuracy: 88.3
    Epoch: 180
    Loss: 1.64020297253
    Training Accuracy: 88.4066666667
    Epoch: 181
    Loss: 1.50509201126
    Training Accuracy: 88.8816666667
    Epoch: 182
    Loss: 1.56817417097
    Training Accuracy: 88.36
    Epoch: 183
    Loss: 1.38196863479
    Training Accuracy: 87.8133333333
    Epoch: 184
    Loss: 1.5552258854
    Training Accuracy: 88.775
    Epoch: 185
    Loss: 1.54264972804
    Training Accuracy: 88.705
    Epoch: 186
    Loss: 1.51080608173
    Training Accuracy: 88.6466666667
    Epoch: 187
    Loss: 1.48761685368
    Training Accuracy: 88.7283333333
    Epoch: 188
    Loss: 1.57839533405
    Training Accuracy: 88.36
    Epoch: 189
    Loss: 1.56343553786
    Training Accuracy: 88.355
    Epoch: 190
    Loss: 1.49730909604
    Training Accuracy: 88.575
    Epoch: 191
    Loss: 1.59373046432
    Training Accuracy: 89.07
    Epoch: 192
    Loss: 1.63377318272
    Training Accuracy: 88.55
    Epoch: 193
    Loss: 1.50038560896
    Training Accuracy: 89.02
    Epoch: 194
    Loss: 1.586914044
    Training Accuracy: 88.4566666667
    Epoch: 195
    Loss: 1.55216493769
    Training Accuracy: 87.935
    Epoch: 196
    Loss: 1.63170811503
    Training Accuracy: 89.0066666667
    Epoch: 197
    Loss: 1.38043054408
    Training Accuracy: 89.105
    Epoch: 198
    Loss: 1.49782733828
    Training Accuracy: 89.04
    Epoch: 199
    Loss: 1.49906733294
    Training Accuracy: 88.4133333333
    Epoch: 200
    Loss: 1.43160671918
    Training Accuracy: 88.5916666667
    Epoch: 201
    Loss: 1.59586355714
    Training Accuracy: 88.83
    Epoch: 202
    Loss: 1.45699385277
    Training Accuracy: 89.18
    Epoch: 203
    Loss: 1.6146873644
    Training Accuracy: 88.41
    Epoch: 204
    Loss: 1.63612013606
    Training Accuracy: 87.895
    Epoch: 205
    Loss: 1.56511449527
    Training Accuracy: 88.05
    Epoch: 206
    Loss: 1.57384165032
    Training Accuracy: 88.7
    Epoch: 207
    Loss: 1.4837942977
    Training Accuracy: 88.855
    Epoch: 208
    Loss: 1.4922999129
    Training Accuracy: 88.5016666667
    Epoch: 209
    Loss: 1.62967450684
    Training Accuracy: 88.65
    Epoch: 210
    Loss: 1.516330897
    Training Accuracy: 88.1083333333
    Epoch: 211
    Loss: 1.51200037331
    Training Accuracy: 89.2583333333
    Epoch: 212
    Loss: 1.48524052885
    Training Accuracy: 89.0233333333
    Epoch: 213
    Loss: 1.49857380213
    Training Accuracy: 88.21
    Epoch: 214
    Loss: 1.53957699844
    Training Accuracy: 88.4533333333
    Epoch: 215
    Loss: 1.53581624681
    Training Accuracy: 88.265
    Epoch: 216
    Loss: 1.45776591613
    Training Accuracy: 89.01
    Epoch: 217
    Loss: 1.54923063343
    Training Accuracy: 88.8566666667
    Epoch: 218
    Loss: 1.52405984907
    Training Accuracy: 89.035
    Epoch: 219
    Loss: 1.63505239361
    Training Accuracy: 88.4433333333
    Epoch: 220
    Loss: 1.44562030278
    Training Accuracy: 88.8883333333
    Epoch: 221
    Loss: 1.43844339375
    Training Accuracy: 89.325
    Epoch: 222
    Loss: 1.4332033572
    Training Accuracy: 89.62
    Epoch: 223
    Loss: 1.35584682986
    Training Accuracy: 89.3766666667
    Epoch: 224
    Loss: 1.39690730964
    Training Accuracy: 88.59
    Epoch: 225
    Loss: 1.55546766707
    Training Accuracy: 87.9216666667
    Epoch: 226
    Loss: 1.51792671001
    Training Accuracy: 88.645
    Epoch: 227
    Loss: 1.50737584108
    Training Accuracy: 89.01
    Epoch: 228
    Loss: 1.47325708201
    Training Accuracy: 88.7216666667
    Epoch: 229
    Loss: 1.35951254035
    Training Accuracy: 89.2
    Epoch: 230
    Loss: 1.51441271898
    Training Accuracy: 88.4616666667
    Epoch: 231
    Loss: 1.49489512274
    Training Accuracy: 88.95
    Epoch: 232
    Loss: 1.51721413425
    Training Accuracy: 88.8366666667
    Epoch: 233
    Loss: 1.41989461423
    Training Accuracy: 88.6916666667
    Epoch: 234
    Loss: 1.46927586467
    Training Accuracy: 89.0266666667
    Epoch: 235
    Loss: 1.53962394814
    Training Accuracy: 88.9316666667
    Epoch: 236
    Loss: 1.35596811945
    Training Accuracy: 88.8833333333
    Epoch: 237
    Loss: 1.45343823317
    Training Accuracy: 88.7133333333
    Epoch: 238
    Loss: 1.53752593247
    Training Accuracy: 89.1283333333
    Epoch: 239
    Loss: 1.46795703137
    Training Accuracy: 89.0816666667
    Epoch: 240
    Loss: 1.44756689251
    Training Accuracy: 89.015
    Epoch: 241
    Loss: 1.54716424105
    Training Accuracy: 89.155
    Epoch: 242
    Loss: 1.44735549683
    Training Accuracy: 88.9366666667
    Epoch: 243
    Loss: 1.47075244619
    Training Accuracy: 88.875
    Epoch: 244
    Loss: 1.42591474285
    Training Accuracy: 88.89
    Epoch: 245
    Loss: 1.46600726837
    Training Accuracy: 88.8183333333
    Epoch: 246
    Loss: 1.46771086397
    Training Accuracy: 88.965
    Epoch: 247
    Loss: 1.37388064239
    Training Accuracy: 88.94
    Epoch: 248
    Loss: 1.35874354676
    Training Accuracy: 88.5266666667
    Epoch: 249
    Loss: 1.45577592227
    Training Accuracy: 89.545
    Epoch: 250
    Loss: 1.46390033938
    Training Accuracy: 88.4
    Epoch: 251
    Loss: 1.47503592353
    Training Accuracy: 88.945
    Epoch: 252
    Loss: 1.43048478705
    Training Accuracy: 89.4333333333
    Epoch: 253
    Loss: 1.4444213296
    Training Accuracy: 88.8983333333
    Epoch: 254
    Loss: 1.29856803503
    Training Accuracy: 88.7566666667
    Epoch: 255
    Loss: 1.37616565028
    Training Accuracy: 89.2566666667
    Epoch: 256
    Loss: 1.48520738056
    Training Accuracy: 89.26
    Epoch: 257
    Loss: 1.48268232053
    Training Accuracy: 89.0716666667
    Epoch: 258
    Loss: 1.41094817065
    Training Accuracy: 89.4216666667
    Epoch: 259
    Loss: 1.54991724016
    Training Accuracy: 89.035
    Epoch: 260
    Loss: 1.52508039583
    Training Accuracy: 88.825
    Epoch: 261
    Loss: 1.46857006328
    Training Accuracy: 88.62
    Epoch: 262
    Loss: 1.39206270638
    Training Accuracy: 88.88
    Epoch: 263
    Loss: 1.46520770078
    Training Accuracy: 88.9666666667
    Epoch: 264
    Loss: 1.36455160024
    Training Accuracy: 89.405
    Epoch: 265
    Loss: 1.4905774188
    Training Accuracy: 88.76
    Epoch: 266
    Loss: 1.39828361242
    Training Accuracy: 89.265
    Epoch: 267
    Loss: 1.39944354567
    Training Accuracy: 89.2533333333
    Epoch: 268
    Loss: 1.31552678757
    Training Accuracy: 88.77
    Epoch: 269
    Loss: 1.50106960662
    Training Accuracy: 89.8466666667
    Epoch: 270
    Loss: 1.41487566745
    Training Accuracy: 89.5133333333
    Epoch: 271
    Loss: 1.31354863075
    Training Accuracy: 89.0016666667
    Epoch: 272
    Loss: 1.44021118492
    Training Accuracy: 89.3183333333
    Epoch: 273
    Loss: 1.38699518753
    Training Accuracy: 88.8683333333
    Epoch: 274
    Loss: 1.32807951422
    Training Accuracy: 89.7333333333
    Epoch: 275
    Loss: 1.39551623111
    Training Accuracy: 89.225
    Epoch: 276
    Loss: 1.36908487049
    Training Accuracy: 89.5916666667
    Epoch: 277
    Loss: 1.48565424398
    Training Accuracy: 89.1066666667
    Epoch: 278
    Loss: 1.37356847816
    Training Accuracy: 89.7166666667
    Epoch: 279
    Loss: 1.37403291946
    Training Accuracy: 89.405
    Epoch: 280
    Loss: 1.48065812319
    Training Accuracy: 89.15
    Epoch: 281
    Loss: 1.36971786138
    Training Accuracy: 89.48
    Epoch: 282
    Loss: 1.30843489829
    Training Accuracy: 89.3633333333
    Epoch: 283
    Loss: 1.43229377519
    Training Accuracy: 88.75
    Epoch: 284
    Loss: 1.35653826969
    Training Accuracy: 89.2066666667
    Epoch: 285
    Loss: 1.3322398772
    Training Accuracy: 89.05
    Epoch: 286
    Loss: 1.42999749398
    Training Accuracy: 89.6783333333
    Epoch: 287
    Loss: 1.41252373119
    Training Accuracy: 89.755
    Epoch: 288
    Loss: 1.35934179944
    Training Accuracy: 89.6983333333
    Epoch: 289
    Loss: 1.25853728875
    Training Accuracy: 89.55
    Epoch: 290
    Loss: 1.26834909896
    Training Accuracy: 89.9266666667
    Epoch: 291
    Loss: 1.27391179905
    Training Accuracy: 89.845
    Epoch: 292
    Loss: 1.38945937115
    Training Accuracy: 89.285
    Epoch: 293
    Loss: 1.42539357086
    Training Accuracy: 89.465
    Epoch: 294
    Loss: 1.33454764337
    Training Accuracy: 89.2516666667
    Epoch: 295
    Loss: 1.28203121856
    Training Accuracy: 89.4233333333
    Epoch: 296
    Loss: 1.31199949705
    Training Accuracy: 88.8716666667
    Epoch: 297
    Loss: 1.28203564139
    Training Accuracy: 88.8266666667
    Epoch: 298
    Loss: 1.36773320519
    Training Accuracy: 89.3516666667
    Epoch: 299
    Loss: 1.33169207438
    Training Accuracy: 89.465
    Epoch: 300
    Loss: 1.35241448432
    Training Accuracy: 89.57
    Epoch: 301
    Loss: 1.31861932605
    Training Accuracy: 89.2766666667
    Epoch: 302
    Loss: 1.32329084488
    Training Accuracy: 90.0316666667
    Epoch: 303
    Loss: 1.38191313117
    Training Accuracy: 89.9433333333
    Epoch: 304
    Loss: 1.30831897393
    Training Accuracy: 89.8633333333
    Epoch: 305
    Loss: 1.42392914169
    Training Accuracy: 89.985
    Epoch: 306
    Loss: 1.31916459323
    Training Accuracy: 89.9533333333
    Epoch: 307
    Loss: 1.2235822616
    Training Accuracy: 89.6183333333
    Epoch: 308
    Loss: 1.31026356554
    Training Accuracy: 89.755
    Epoch: 309
    Loss: 1.27513081543
    Training Accuracy: 89.5416666667
    Epoch: 310
    Loss: 1.27439939709
    Training Accuracy: 89.8016666667
    Epoch: 311
    Loss: 1.27062331631
    Training Accuracy: 89.8833333333
    Epoch: 312
    Loss: 1.32667579817
    Training Accuracy: 89.9133333333
    Epoch: 313
    Loss: 1.26023533792
    Training Accuracy: 89.5
    Epoch: 314
    Loss: 1.31270228486
    Training Accuracy: 90.015
    Epoch: 315
    Loss: 1.3121642029
    Training Accuracy: 89.8233333333
    Epoch: 316
    Loss: 1.29462040478
    Training Accuracy: 90.245
    Epoch: 317
    Loss: 1.24576001328
    Training Accuracy: 90.17
    Epoch: 318
    Loss: 1.22351694569
    Training Accuracy: 89.7066666667
    Epoch: 319
    Loss: 1.35480199954
    Training Accuracy: 89.98
    Epoch: 320
    Loss: 1.35122487956
    Training Accuracy: 89.4083333333
    Epoch: 321
    Loss: 1.39825060945
    Training Accuracy: 89.6166666667
    Epoch: 322
    Loss: 1.30796047153
    Training Accuracy: 89.875
    Epoch: 323
    Loss: 1.28458602641
    Training Accuracy: 89.6366666667
    Epoch: 324
    Loss: 1.32365046184
    Training Accuracy: 89.8983333333
    Epoch: 325
    Loss: 1.25612493103
    Training Accuracy: 89.3633333333
    Epoch: 326
    Loss: 1.21959556295
    Training Accuracy: 89.9166666667
    Epoch: 327
    Loss: 1.27131887373
    Training Accuracy: 90.1816666667
    Epoch: 328
    Loss: 1.3228704021
    Training Accuracy: 90.2616666667
    Epoch: 329
    Loss: 1.28548811012
    Training Accuracy: 89.6566666667
    Epoch: 330
    Loss: 1.26219207059
    Training Accuracy: 89.635
    Epoch: 331
    Loss: 1.29787725507
    Training Accuracy: 89.8433333333
    Epoch: 332
    Loss: 1.29512153446
    Training Accuracy: 90.1083333333
    Epoch: 333
    Loss: 1.25023843836
    Training Accuracy: 90.3133333333
    Epoch: 334
    Loss: 1.28356219463
    Training Accuracy: 90.4716666667
    Epoch: 335
    Loss: 1.31535723286
    Training Accuracy: 90.21
    Epoch: 336
    Loss: 1.16986970291
    Training Accuracy: 90.025
    Epoch: 337
    Loss: 1.21248848904
    Training Accuracy: 90.0033333333
    Epoch: 338
    Loss: 1.26924794317
    Training Accuracy: 90.3366666667
    Epoch: 339
    Loss: 1.19128005956
    Training Accuracy: 90.0216666667
    Epoch: 340
    Loss: 1.21480973115
    Training Accuracy: 89.8366666667
    Epoch: 341
    Loss: 1.17431253557
    Training Accuracy: 90.0866666667
    Epoch: 342
    Loss: 1.28520823638
    Training Accuracy: 90.0166666667
    Epoch: 343
    Loss: 1.34699566395
    Training Accuracy: 90.37
    Epoch: 344
    Loss: 1.14959627434
    Training Accuracy: 90.2333333333
    Epoch: 345
    Loss: 1.23805847246
    Training Accuracy: 90.36
    Epoch: 346
    Loss: 1.21997119389
    Training Accuracy: 90.0033333333
    Epoch: 347
    Loss: 1.24100758758
    Training Accuracy: 90.4883333333
    Epoch: 348
    Loss: 1.21596600417
    Training Accuracy: 89.985
    Epoch: 349
    Loss: 1.25971249496
    Training Accuracy: 90.4283333333
    Epoch: 350
    Loss: 1.17432941435
    Training Accuracy: 89.9133333333
    Epoch: 351
    Loss: 1.21678999183
    Training Accuracy: 89.795
    Epoch: 352
    Loss: 1.18748965824
    Training Accuracy: 90.5466666667
    Epoch: 353
    Loss: 1.11765379351
    Training Accuracy: 90.2733333333
    Epoch: 354
    Loss: 1.23361490533
    Training Accuracy: 90.3483333333
    Epoch: 355
    Loss: 1.2243665187
    Training Accuracy: 90.405
    Epoch: 356
    Loss: 1.1825797505
    Training Accuracy: 90.2566666667
    Epoch: 357
    Loss: 1.17570357041
    Training Accuracy: 90.0083333333
    Epoch: 358
    Loss: 1.16538142895
    Training Accuracy: 90.1066666667
    Epoch: 359
    Loss: 1.14761248021
    Training Accuracy: 90.33
    Epoch: 360
    Loss: 1.10853357456
    Training Accuracy: 90.7416666667
    Epoch: 361
    Loss: 1.26272136788
    Training Accuracy: 90.3483333333
    Epoch: 362
    Loss: 1.1667303604
    Training Accuracy: 89.6
    Epoch: 363
    Loss: 1.25048260999
    Training Accuracy: 90.28
    Epoch: 364
    Loss: 1.15917586377
    Training Accuracy: 90.6983333333
    Epoch: 365
    Loss: 1.29693893513
    Training Accuracy: 89.9316666667
    Epoch: 366
    Loss: 1.10441824681
    Training Accuracy: 90.7683333333
    Epoch: 367
    Loss: 1.22832395188
    Training Accuracy: 90.1416666667
    Epoch: 368
    Loss: 1.16754711758
    Training Accuracy: 90.0183333333
    Epoch: 369
    Loss: 1.2377489166
    Training Accuracy: 90.6033333333
    Epoch: 370
    Loss: 1.14798482623
    Training Accuracy: 90.6216666667
    Epoch: 371
    Loss: 1.1272868398
    Training Accuracy: 90.3066666667
    Epoch: 372
    Loss: 1.29389837795
    Training Accuracy: 90.385
    Epoch: 373
    Loss: 1.05971834159
    Training Accuracy: 90.3883333333
    Epoch: 374
    Loss: 1.11216122955
    Training Accuracy: 90.415
    Epoch: 375
    Loss: 1.27331169175
    Training Accuracy: 90.24
    Epoch: 376
    Loss: 1.18176691971
    Training Accuracy: 90.5066666667
    Epoch: 377
    Loss: 1.05907196076
    Training Accuracy: 90.6983333333
    Epoch: 378
    Loss: 0.986857402598
    Training Accuracy: 90.3616666667
    Epoch: 379
    Loss: 1.07616091581
    Training Accuracy: 90.78
    Epoch: 380
    Loss: 1.19101462327
    Training Accuracy: 90.7933333333
    Epoch: 381
    Loss: 1.18747091094
    Training Accuracy: 90.6916666667
    Epoch: 382
    Loss: 1.1290360191
    Training Accuracy: 90.915
    Epoch: 383
    Loss: 1.15663107577
    Training Accuracy: 90.9316666667
    Epoch: 384
    Loss: 1.04123780459
    Training Accuracy: 90.3683333333
    Epoch: 385
    Loss: 1.07054852541
    Training Accuracy: 90.4233333333
    Epoch: 386
    Loss: 1.17746120839
    Training Accuracy: 90.595
    Epoch: 387
    Loss: 1.03718898855
    Training Accuracy: 90.485
    Epoch: 388
    Loss: 1.13735122028
    Training Accuracy: 90.46
    Epoch: 389
    Loss: 1.00917184529
    Training Accuracy: 90.5066666667
    Epoch: 390
    Loss: 1.129504305
    Training Accuracy: 90.585
    Epoch: 391
    Loss: 1.15294780369
    Training Accuracy: 90.8033333333
    Epoch: 392
    Loss: 1.20549573575
    Training Accuracy: 90.6333333333
    Epoch: 393
    Loss: 1.18041387457
    Training Accuracy: 90.2466666667
    Epoch: 394
    Loss: 0.962557398955
    Training Accuracy: 90.33
    Epoch: 395
    Loss: 1.12790245233
    Training Accuracy: 90.8983333333
    Epoch: 396
    Loss: 1.05228594235
    Training Accuracy: 90.8116666667
    Epoch: 397
    Loss: 1.0512646371
    Training Accuracy: 90.4316666667
    Epoch: 398
    Loss: 1.23392420842
    Training Accuracy: 90.98
    Epoch: 399
    Loss: 1.07160200681
    Training Accuracy: 90.6716666667
    Epoch: 400
    Loss: 1.11728225028
    Training Accuracy: 90.7133333333
    Epoch: 401
    Loss: 1.10568480453
    Training Accuracy: 90.86
    Epoch: 402
    Loss: 1.09239445442
    Training Accuracy: 90.7483333333
    Epoch: 403
    Loss: 1.00616849778
    Training Accuracy: 90.785
    Epoch: 404
    Loss: 1.05255729082
    Training Accuracy: 90.5866666667
    Epoch: 405
    Loss: 1.16316797429
    Training Accuracy: 90.9566666667
    Epoch: 406
    Loss: 1.0781721155
    Training Accuracy: 90.7233333333
    Epoch: 407
    Loss: 1.06826103637
    Training Accuracy: 90.6866666667
    Epoch: 408
    Loss: 1.09867656206
    Training Accuracy: 91.27
    Epoch: 409
    Loss: 1.18991338398
    Training Accuracy: 90.6766666667
    Epoch: 410
    Loss: 1.0575848008
    Training Accuracy: 90.8066666667
    Epoch: 411
    Loss: 0.993276644429
    Training Accuracy: 90.8016666667
    Epoch: 412
    Loss: 1.07426560167
    Training Accuracy: 90.8383333333
    Epoch: 413
    Loss: 1.19161083019
    Training Accuracy: 91.2783333333
    Epoch: 414
    Loss: 1.02837942753
    Training Accuracy: 90.5633333333
    Epoch: 415
    Loss: 1.04595110375
    Training Accuracy: 91.1083333333
    Epoch: 416
    Loss: 1.08159979175
    Training Accuracy: 91.0783333333
    Epoch: 417
    Loss: 1.0535712097
    Training Accuracy: 90.9633333333
    Epoch: 418
    Loss: 1.04412879296
    Training Accuracy: 91.2516666667
    Epoch: 419
    Loss: 1.08507567899
    Training Accuracy: 90.95
    Epoch: 420
    Loss: 1.0822524042
    Training Accuracy: 90.8616666667
    Epoch: 421
    Loss: 1.08421697352
    Training Accuracy: 90.8016666667
    Epoch: 422
    Loss: 1.00502799009
    Training Accuracy: 90.8983333333
    Epoch: 423
    Loss: 1.03951918164
    Training Accuracy: 90.9316666667
    Epoch: 424
    Loss: 1.00332440035
    Training Accuracy: 91.0833333333
    Epoch: 425
    Loss: 1.01284419933
    Training Accuracy: 91.1266666667
    Epoch: 426
    Loss: 1.02266819808
    Training Accuracy: 90.99
    Epoch: 427
    Loss: 1.18842366159
    Training Accuracy: 91.0
    Epoch: 428
    Loss: 0.976532375156
    Training Accuracy: 90.405
    Epoch: 429
    Loss: 0.979453899231
    Training Accuracy: 90.8533333333
    Epoch: 430
    Loss: 1.04911579061
    Training Accuracy: 90.83
    Epoch: 431
    Loss: 0.97350653322
    Training Accuracy: 91.0466666667
    Epoch: 432
    Loss: 0.947227734651
    Training Accuracy: 91.2183333333
    Epoch: 433
    Loss: 0.965621621765
    Training Accuracy: 91.5266666667
    Epoch: 434
    Loss: 0.991874586206
    Training Accuracy: 91.0316666667
    Epoch: 435
    Loss: 0.996046340853
    Training Accuracy: 91.1533333333
    Epoch: 436
    Loss: 0.998264200198
    Training Accuracy: 91.2833333333
    Epoch: 437
    Loss: 0.977902960639
    Training Accuracy: 91.6266666667
    Epoch: 438
    Loss: 0.975000631574
    Training Accuracy: 91.665
    Epoch: 439
    Loss: 0.973059806361
    Training Accuracy: 91.4083333333
    Epoch: 440
    Loss: 1.04087709316
    Training Accuracy: 91.3566666667
    Epoch: 441
    Loss: 1.00881808909
    Training Accuracy: 91.525
    Epoch: 442
    Loss: 0.906488876048
    Training Accuracy: 91.4166666667
    Epoch: 443
    Loss: 0.974192918341
    Training Accuracy: 91.6083333333
    Epoch: 444
    Loss: 0.967724713342
    Training Accuracy: 91.3616666667
    Epoch: 445
    Loss: 1.12043147057
    Training Accuracy: 91.715
    Epoch: 446
    Loss: 0.955132203193
    Training Accuracy: 91.3683333333
    Epoch: 447
    Loss: 1.02835772555
    Training Accuracy: 91.1666666667
    Epoch: 448
    Loss: 1.11972286391
    Training Accuracy: 91.0816666667
    Epoch: 449
    Loss: 1.02229793908
    Training Accuracy: 91.6016666667
    Epoch: 450
    Loss: 1.04468644873
    Training Accuracy: 91.285
    Epoch: 451
    Loss: 0.944512622866
    Training Accuracy: 91.585
    Epoch: 452
    Loss: 1.00860608251
    Training Accuracy: 91.685
    Epoch: 453
    Loss: 0.938552756659
    Training Accuracy: 91.5283333333
    Epoch: 454
    Loss: 0.895407697206
    Training Accuracy: 91.4166666667
    Epoch: 455
    Loss: 0.910661266249
    Training Accuracy: 91.6516666667
    Epoch: 456
    Loss: 0.962566894755
    Training Accuracy: 91.3166666667
    Epoch: 457
    Loss: 0.989750933112
    Training Accuracy: 91.7083333333
    Epoch: 458
    Loss: 1.03527026244
    Training Accuracy: 91.26
    Epoch: 459
    Loss: 0.94622987744
    Training Accuracy: 91.3216666667
    Epoch: 460
    Loss: 0.919009564309
    Training Accuracy: 91.4316666667
    Epoch: 461
    Loss: 0.998965434716
    Training Accuracy: 91.545
    Epoch: 462
    Loss: 0.902888859196
    Training Accuracy: 91.775
    Epoch: 463
    Loss: 1.02987656295
    Training Accuracy: 91.58
    Epoch: 464
    Loss: 0.949360104728
    Training Accuracy: 92.0333333333
    Epoch: 465
    Loss: 1.08210448482
    Training Accuracy: 91.5833333333
    Epoch: 466
    Loss: 1.0360140348
    Training Accuracy: 91.7516666667
    Epoch: 467
    Loss: 0.928897558222
    Training Accuracy: 91.7116666667
    Epoch: 468
    Loss: 0.930886626392
    Training Accuracy: 91.6616666667
    Epoch: 469
    Loss: 1.0294612768
    Training Accuracy: 91.4583333333
    Epoch: 470
    Loss: 0.939159021033
    Training Accuracy: 91.6916666667
    Epoch: 471
    Loss: 0.941825045225
    Training Accuracy: 91.59
    Epoch: 472
    Loss: 0.959120854629
    Training Accuracy: 91.5416666667
    Epoch: 473
    Loss: 0.964871708116
    Training Accuracy: 91.7983333333
    Epoch: 474
    Loss: 0.910360466399
    Training Accuracy: 91.87
    Epoch: 475
    Loss: 0.994323623706
    Training Accuracy: 91.6666666667
    Epoch: 476
    Loss: 0.969673532506
    Training Accuracy: 91.73
    Epoch: 477
    Loss: 0.95130716401
    Training Accuracy: 91.7183333333
    Epoch: 478
    Loss: 0.952835083948
    Training Accuracy: 91.4633333333
    Epoch: 479
    Loss: 0.923309627285
    Training Accuracy: 91.86
    Epoch: 480
    Loss: 1.02759464259
    Training Accuracy: 91.7716666667
    Epoch: 481
    Loss: 0.898359549732
    Training Accuracy: 91.8633333333
    Epoch: 482
    Loss: 0.924828185971
    Training Accuracy: 92.0366666667
    Epoch: 483
    Loss: 0.898786714065
    Training Accuracy: 91.76
    Epoch: 484
    Loss: 0.871393415606
    Training Accuracy: 91.6383333333
    Epoch: 485
    Loss: 0.909381941579
    Training Accuracy: 92.1233333333
    Epoch: 486
    Loss: 0.824631410946
    Training Accuracy: 92.0966666667
    Epoch: 487
    Loss: 1.01601937441
    Training Accuracy: 92.0266666667
    Epoch: 488
    Loss: 1.03485377741
    Training Accuracy: 91.825
    Epoch: 489
    Loss: 0.921912544946
    Training Accuracy: 92.245
    Epoch: 490
    Loss: 0.895582208691
    Training Accuracy: 91.7916666667
    Epoch: 491
    Loss: 0.91593181964
    Training Accuracy: 92.0216666667
    Epoch: 492
    Loss: 0.817485371182
    Training Accuracy: 91.8633333333
    Epoch: 493
    Loss: 0.924638758598
    Training Accuracy: 92.06
    Epoch: 494
    Loss: 0.953807254419
    Training Accuracy: 91.9733333333
    Epoch: 495
    Loss: 0.918108093907
    Training Accuracy: 92.31
    Epoch: 496
    Loss: 0.80053539917
    Training Accuracy: 92.1066666667
    Epoch: 497
    Loss: 0.89553559538
    Training Accuracy: 92.315
    Epoch: 498
    Loss: 0.795739335183
    Training Accuracy: 91.5516666667
    Epoch: 499
    Loss: 0.917460033508
    Training Accuracy: 91.8483333333
    Epoch: 500
    Loss: 0.835293788914
    Training Accuracy: 91.94
    Epoch: 501
    Loss: 0.796222280422
    Training Accuracy: 92.095
    Epoch: 502
    Loss: 0.898525817251
    Training Accuracy: 91.9866666667
    Epoch: 503
    Loss: 0.864929857602
    Training Accuracy: 92.1033333333
    Epoch: 504
    Loss: 0.865331965183
    Training Accuracy: 92.3883333333
    Epoch: 505
    Loss: 0.839247630822
    Training Accuracy: 92.3733333333
    Epoch: 506
    Loss: 0.786912703778
    Training Accuracy: 92.2166666667
    Epoch: 507
    Loss: 0.863718204802
    Training Accuracy: 92.3633333333
    Epoch: 508
    Loss: 0.834160623164
    Training Accuracy: 92.3716666667
    Epoch: 509
    Loss: 0.820292849063
    Training Accuracy: 92.2433333333
    Epoch: 510
    Loss: 0.970252598537
    Training Accuracy: 92.3583333333
    Epoch: 511
    Loss: 0.776089096168
    Training Accuracy: 92.3116666667
    Epoch: 512
    Loss: 0.875997545316
    Training Accuracy: 92.4533333333
    Epoch: 513
    Loss: 0.878214548887
    Training Accuracy: 92.2983333333
    Epoch: 514
    Loss: 0.958534907166
    Training Accuracy: 92.495
    Epoch: 515
    Loss: 0.778666626424
    Training Accuracy: 92.37
    Epoch: 516
    Loss: 0.923045713645
    Training Accuracy: 92.2483333333
    Epoch: 517
    Loss: 0.830784688259
    Training Accuracy: 91.8466666667
    Epoch: 518
    Loss: 0.847758100156
    Training Accuracy: 92.315
    Epoch: 519
    Loss: 0.856388581851
    Training Accuracy: 92.0533333333
    Epoch: 520
    Loss: 0.815902710997
    Training Accuracy: 92.0116666667
    Epoch: 521
    Loss: 0.916499660702
    Training Accuracy: 92.27
    Epoch: 522
    Loss: 0.858502005913
    Training Accuracy: 92.3516666667
    Epoch: 523
    Loss: 0.701911831667
    Training Accuracy: 92.3633333333
    Epoch: 524
    Loss: 0.909542392998
    Training Accuracy: 92.38
    Epoch: 525
    Loss: 0.788792586099
    Training Accuracy: 92.4083333333
    Epoch: 526
    Loss: 0.798825933263
    Training Accuracy: 92.4766666667
    Epoch: 527
    Loss: 0.776477125627
    Training Accuracy: 92.4933333333
    Epoch: 528
    Loss: 0.794113824736
    Training Accuracy: 92.58
    Epoch: 529
    Loss: 0.888090329093
    Training Accuracy: 92.48
    Epoch: 530
    Loss: 0.763514962922
    Training Accuracy: 92.445
    Epoch: 531
    Loss: 0.88048479245
    Training Accuracy: 92.465
    Epoch: 532
    Loss: 0.911095536703
    Training Accuracy: 92.3033333333
    Epoch: 533
    Loss: 0.820502905276
    Training Accuracy: 92.58
    Epoch: 534
    Loss: 0.769601218746
    Training Accuracy: 92.6
    Epoch: 535
    Loss: 0.824199809864
    Training Accuracy: 92.4966666667
    Epoch: 536
    Loss: 0.768626494456
    Training Accuracy: 92.7766666667
    Epoch: 537
    Loss: 0.741292547515
    Training Accuracy: 92.495
    Epoch: 538
    Loss: 0.720126630888
    Training Accuracy: 92.4583333333
    Epoch: 539
    Loss: 0.778742242052
    Training Accuracy: 92.5583333333
    Epoch: 540
    Loss: 0.74912801436
    Training Accuracy: 92.7333333333
    Epoch: 541
    Loss: 0.855263303359
    Training Accuracy: 92.7633333333
    Epoch: 542
    Loss: 0.787387139092
    Training Accuracy: 92.685
    Epoch: 543
    Loss: 0.856501758428
    Training Accuracy: 92.5916666667
    Epoch: 544
    Loss: 0.760723442141
    Training Accuracy: 92.685
    Epoch: 545
    Loss: 0.801813257362
    Training Accuracy: 92.7666666667
    Epoch: 546
    Loss: 0.86113736359
    Training Accuracy: 92.365
    Epoch: 547
    Loss: 0.848022099431
    Training Accuracy: 92.7433333333
    Epoch: 548
    Loss: 0.825586781009
    Training Accuracy: 92.7983333333
    Epoch: 549
    Loss: 0.851254692307
    Training Accuracy: 92.765
    Epoch: 550
    Loss: 0.752920915427
    Training Accuracy: 92.695
    Epoch: 551
    Loss: 0.871431945015
    Training Accuracy: 92.9133333333
    Epoch: 552
    Loss: 0.758166102698
    Training Accuracy: 92.7433333333
    Epoch: 553
    Loss: 0.719786945666
    Training Accuracy: 92.7166666667
    Epoch: 554
    Loss: 0.793484521575
    Training Accuracy: 92.7666666667
    Epoch: 555
    Loss: 0.811012294709
    Training Accuracy: 92.5233333333
    Epoch: 556
    Loss: 0.823922453907
    Training Accuracy: 92.9583333333
    Epoch: 557
    Loss: 0.755155721956
    Training Accuracy: 92.985
    Epoch: 558
    Loss: 0.826613689083
    Training Accuracy: 92.7883333333
    Epoch: 559
    Loss: 0.792073536957
    Training Accuracy: 92.9916666667
    Epoch: 560
    Loss: 0.889246911139
    Training Accuracy: 93.1
    Epoch: 561
    Loss: 0.779584297435
    Training Accuracy: 93.0283333333
    Epoch: 562
    Loss: 0.852283068816
    Training Accuracy: 92.715
    Epoch: 563
    Loss: 0.780732673921
    Training Accuracy: 92.7333333333
    Epoch: 564
    Loss: 0.752933066693
    Training Accuracy: 92.755
    Epoch: 565
    Loss: 0.787356440828
    Training Accuracy: 93.0466666667
    Epoch: 566
    Loss: 0.700880671462
    Training Accuracy: 93.0566666667
    Epoch: 567
    Loss: 0.74492990249
    Training Accuracy: 93.0633333333
    Epoch: 568
    Loss: 0.750633216033
    Training Accuracy: 92.795
    Epoch: 569
    Loss: 0.734718560843
    Training Accuracy: 92.9733333333
    Epoch: 570
    Loss: 0.795391293051
    Training Accuracy: 92.9216666667
    Epoch: 571
    Loss: 0.73230036811
    Training Accuracy: 93.0766666667
    Epoch: 572
    Loss: 0.744330141114
    Training Accuracy: 93.0416666667
    Epoch: 573
    Loss: 0.718394935854
    Training Accuracy: 92.98
    Epoch: 574
    Loss: 0.629415829964
    Training Accuracy: 92.9533333333
    Epoch: 575
    Loss: 0.688431128724
    Training Accuracy: 93.1833333333
    Epoch: 576
    Loss: 0.765015819971
    Training Accuracy: 92.9733333333
    Epoch: 577
    Loss: 0.744344309251
    Training Accuracy: 93.015
    Epoch: 578
    Loss: 0.797794013875
    Training Accuracy: 93.1083333333
    Epoch: 579
    Loss: 0.737748019828
    Training Accuracy: 92.9766666667
    Epoch: 580
    Loss: 0.706750885289
    Training Accuracy: 93.105
    Epoch: 581
    Loss: 0.787031891724
    Training Accuracy: 93.2283333333
    Epoch: 582
    Loss: 0.731526627433
    Training Accuracy: 93.0733333333
    Epoch: 583
    Loss: 0.729052221808
    Training Accuracy: 93.06
    Epoch: 584
    Loss: 0.751017185052
    Training Accuracy: 92.9133333333
    Epoch: 585
    Loss: 0.658160970474
    Training Accuracy: 93.2466666667
    Epoch: 586
    Loss: 0.728392134999
    Training Accuracy: 92.9933333333
    Epoch: 587
    Loss: 0.731635344436
    Training Accuracy: 93.14
    Epoch: 588
    Loss: 0.658705446293
    Training Accuracy: 93.2433333333
    Epoch: 589
    Loss: 0.780148798494
    Training Accuracy: 93.2233333333
    Epoch: 590
    Loss: 0.784996547653
    Training Accuracy: 93.1283333333
    Epoch: 591
    Loss: 0.724499913063
    Training Accuracy: 92.9366666667
    Epoch: 592
    Loss: 0.839716050043
    Training Accuracy: 93.3666666667
    Epoch: 593
    Loss: 0.693606042787
    Training Accuracy: 93.2766666667
    Epoch: 594
    Loss: 0.61950193624
    Training Accuracy: 93.2533333333
    Epoch: 595
    Loss: 0.711860192608
    Training Accuracy: 93.0266666667
    Epoch: 596
    Loss: 0.677833374116
    Training Accuracy: 93.375
    Epoch: 597
    Loss: 0.670980175951
    Training Accuracy: 93.335
    Epoch: 598
    Loss: 0.665763415086
    Training Accuracy: 93.185
    Epoch: 599
    Loss: 0.771345506738
    Training Accuracy: 93.385
    Epoch: 600
    Loss: 0.706937341042
    Training Accuracy: 93.3583333333
    Epoch: 601
    Loss: 0.767063026569
    Training Accuracy: 93.42
    Epoch: 602
    Loss: 0.657178474084
    Training Accuracy: 93.59
    Epoch: 603
    Loss: 0.629854122914
    Training Accuracy: 93.5166666667
    Epoch: 604
    Loss: 0.689951431814
    Training Accuracy: 93.3183333333
    Epoch: 605
    Loss: 0.767898584568
    Training Accuracy: 93.4216666667
    Epoch: 606
    Loss: 0.788139238545
    Training Accuracy: 93.4466666667
    Epoch: 607
    Loss: 0.686884739283
    Training Accuracy: 93.4733333333
    Epoch: 608
    Loss: 0.682267467548
    Training Accuracy: 93.445
    Epoch: 609
    Loss: 0.63909052569
    Training Accuracy: 93.4933333333
    Epoch: 610
    Loss: 0.611075444054
    Training Accuracy: 93.3516666667
    Epoch: 611
    Loss: 0.619183247521
    Training Accuracy: 93.395
    Epoch: 612
    Loss: 0.705662963282
    Training Accuracy: 93.5683333333
    Epoch: 613
    Loss: 0.689796744099
    Training Accuracy: 93.5683333333
    Epoch: 614
    Loss: 0.702715187759
    Training Accuracy: 93.4533333333
    Epoch: 615
    Loss: 0.689808416164
    Training Accuracy: 93.56
    Epoch: 616
    Loss: 0.66339696261
    Training Accuracy: 93.6066666667
    Epoch: 617
    Loss: 0.773929437373
    Training Accuracy: 93.58
    Epoch: 618
    Loss: 0.648803966806
    Training Accuracy: 93.5716666667
    Epoch: 619
    Loss: 0.629295041392
    Training Accuracy: 93.615
    Epoch: 620
    Loss: 0.689536526403
    Training Accuracy: 93.7833333333
    Epoch: 621
    Loss: 0.55498205187
    Training Accuracy: 93.4433333333
    Epoch: 622
    Loss: 0.638130428888
    Training Accuracy: 93.8016666667
    Epoch: 623
    Loss: 0.733153912565
    Training Accuracy: 93.605
    Epoch: 624
    Loss: 0.683745832569
    Training Accuracy: 93.5083333333
    Epoch: 625
    Loss: 0.649280053193
    Training Accuracy: 93.6016666667
    Epoch: 626
    Loss: 0.719548107438
    Training Accuracy: 93.7316666667
    Epoch: 627
    Loss: 0.621046806569
    Training Accuracy: 93.64
    Epoch: 628
    Loss: 0.649833794924
    Training Accuracy: 93.685
    Epoch: 629
    Loss: 0.600204387676
    Training Accuracy: 93.5066666667
    Epoch: 630
    Loss: 0.688465905269
    Training Accuracy: 93.5366666667
    Epoch: 631
    Loss: 0.586416068128
    Training Accuracy: 93.6633333333
    Epoch: 632
    Loss: 0.689260314401
    Training Accuracy: 93.8983333333
    Epoch: 633
    Loss: 0.707971704829
    Training Accuracy: 93.6866666667
    Epoch: 634
    Loss: 0.601236789247
    Training Accuracy: 93.625
    Epoch: 635
    Loss: 0.586691082011
    Training Accuracy: 93.5466666667
    Epoch: 636
    Loss: 0.687439646724
    Training Accuracy: 93.4166666667
    Epoch: 637
    Loss: 0.609761425351
    Training Accuracy: 93.5383333333
    Epoch: 638
    Loss: 0.695726178296
    Training Accuracy: 93.7366666667
    Epoch: 639
    Loss: 0.635770307009
    Training Accuracy: 93.6116666667
    Epoch: 640
    Loss: 0.722006098769
    Training Accuracy: 93.585
    Epoch: 641
    Loss: 0.649713935833
    Training Accuracy: 93.7133333333
    Epoch: 642
    Loss: 0.630054811588
    Training Accuracy: 93.7933333333
    Epoch: 643
    Loss: 0.735963156423
    Training Accuracy: 93.755
    Epoch: 644
    Loss: 0.677621962063
    Training Accuracy: 93.6266666667
    Epoch: 645
    Loss: 0.687976604335
    Training Accuracy: 93.755
    Epoch: 646
    Loss: 0.63513665101
    Training Accuracy: 93.7766666667
    Epoch: 647
    Loss: 0.650778312048
    Training Accuracy: 93.7616666667
    Epoch: 648
    Loss: 0.6368258522
    Training Accuracy: 93.7916666667
    Epoch: 649
    Loss: 0.618644504571
    Training Accuracy: 93.6633333333
    Epoch: 650
    Loss: 0.611400630368
    Training Accuracy: 93.8216666667
    Epoch: 651
    Loss: 0.577912819377
    Training Accuracy: 93.8133333333
    Epoch: 652
    Loss: 0.582158271344
    Training Accuracy: 93.7016666667
    Epoch: 653
    Loss: 0.659512414194
    Training Accuracy: 93.895
    Epoch: 654
    Loss: 0.623925028801
    Training Accuracy: 93.8666666667
    Epoch: 655
    Loss: 0.639743337789
    Training Accuracy: 93.6833333333
    Epoch: 656
    Loss: 0.648350322312
    Training Accuracy: 93.7966666667
    Epoch: 657
    Loss: 0.644236786268
    Training Accuracy: 93.85
    Epoch: 658
    Loss: 0.630752049296
    Training Accuracy: 93.855
    Epoch: 659
    Loss: 0.599214580083
    Training Accuracy: 93.7166666667
    Epoch: 660
    Loss: 0.564202944514
    Training Accuracy: 93.7333333333
    Epoch: 661
    Loss: 0.632904853724
    Training Accuracy: 93.8066666667
    Epoch: 662
    Loss: 0.487178237386
    Training Accuracy: 93.9016666667
    Epoch: 663
    Loss: 0.632545934274
    Training Accuracy: 93.8183333333
    Epoch: 664
    Loss: 0.64668447676
    Training Accuracy: 93.9016666667
    Epoch: 665
    Loss: 0.580911058692
    Training Accuracy: 94.05
    Epoch: 666
    Loss: 0.557431914396
    Training Accuracy: 93.9633333333
    Epoch: 667
    Loss: 0.642778506243
    Training Accuracy: 94.0783333333
    Epoch: 668
    Loss: 0.592077934877
    Training Accuracy: 94.1183333333
    Epoch: 669
    Loss: 0.610721719021
    Training Accuracy: 94.105
    Epoch: 670
    Loss: 0.619197238381
    Training Accuracy: 94.0616666667
    Epoch: 671
    Loss: 0.610390625829
    Training Accuracy: 93.9016666667
    Epoch: 672
    Loss: 0.540279386191
    Training Accuracy: 94.05
    Epoch: 673
    Loss: 0.572779595738
    Training Accuracy: 94.0766666667
    Epoch: 674
    Loss: 0.587295599287
    Training Accuracy: 93.98
    Epoch: 675
    Loss: 0.58901096497
    Training Accuracy: 94.07
    Epoch: 676
    Loss: 0.554154484616
    Training Accuracy: 93.9816666667
    Epoch: 677
    Loss: 0.527118003581
    Training Accuracy: 94.2616666667
    Epoch: 678
    Loss: 0.549283379123
    Training Accuracy: 94.1366666667
    Epoch: 679
    Loss: 0.598375607854
    Training Accuracy: 93.895
    Epoch: 680
    Loss: 0.532297691115
    Training Accuracy: 94.0416666667
    Epoch: 681
    Loss: 0.580487262126
    Training Accuracy: 94.08
    Epoch: 682
    Loss: 0.6490175988
    Training Accuracy: 94.1866666667
    Epoch: 683
    Loss: 0.595661844157
    Training Accuracy: 94.2416666667
    Epoch: 684
    Loss: 0.528630972163
    Training Accuracy: 94.1483333333
    Epoch: 685
    Loss: 0.641812268844
    Training Accuracy: 94.1
    Epoch: 686
    Loss: 0.660244905546
    Training Accuracy: 94.09
    Epoch: 687
    Loss: 0.549457842506
    Training Accuracy: 94.1016666667
    Epoch: 688
    Loss: 0.626304721476
    Training Accuracy: 94.0866666667
    Epoch: 689
    Loss: 0.516942779927
    Training Accuracy: 94.03
    Epoch: 690
    Loss: 0.606009555482
    Training Accuracy: 94.0066666667
    Epoch: 691
    Loss: 0.583302374141
    Training Accuracy: 94.14
    Epoch: 692
    Loss: 0.63958593913
    Training Accuracy: 94.21
    Epoch: 693
    Loss: 0.559994539355
    Training Accuracy: 94.2066666667
    Epoch: 694
    Loss: 0.577794801401
    Training Accuracy: 94.1216666667
    Epoch: 695
    Loss: 0.566566244371
    Training Accuracy: 94.2966666667
    Epoch: 696
    Loss: 0.546843066244
    Training Accuracy: 94.08
    Epoch: 697
    Loss: 0.557287697566
    Training Accuracy: 94.3066666667
    Epoch: 698
    Loss: 0.494067848149
    Training Accuracy: 94.195
    Epoch: 699
    Loss: 0.516769059656
    Training Accuracy: 94.37
    Epoch: 700
    Loss: 0.642099818315
    Training Accuracy: 94.3083333333
    Epoch: 701
    Loss: 0.599582443496
    Training Accuracy: 94.3083333333
    Epoch: 702
    Loss: 0.575483390365
    Training Accuracy: 94.2633333333
    Epoch: 703
    Loss: 0.485482842374
    Training Accuracy: 94.275
    Epoch: 704
    Loss: 0.567489026866
    Training Accuracy: 94.4
    Epoch: 705
    Loss: 0.523857094082
    Training Accuracy: 94.3516666667
    Epoch: 706
    Loss: 0.548694120852
    Training Accuracy: 94.3516666667
    Epoch: 707
    Loss: 0.527469297085
    Training Accuracy: 94.3033333333
    Epoch: 708
    Loss: 0.538152532622
    Training Accuracy: 94.2683333333
    Epoch: 709
    Loss: 0.489480243477
    Training Accuracy: 94.26
    Epoch: 710
    Loss: 0.629653511454
    Training Accuracy: 94.49
    Epoch: 711
    Loss: 0.608147162898
    Training Accuracy: 94.5133333333
    Epoch: 712
    Loss: 0.579497397658
    Training Accuracy: 94.3766666667
    Epoch: 713
    Loss: 0.538586031121
    Training Accuracy: 94.3816666667
    Epoch: 714
    Loss: 0.499248416377
    Training Accuracy: 94.4466666667
    Epoch: 715
    Loss: 0.560937421798
    Training Accuracy: 94.4383333333
    Epoch: 716
    Loss: 0.48456646755
    Training Accuracy: 94.47
    Epoch: 717
    Loss: 0.54229879015
    Training Accuracy: 94.39
    Epoch: 718
    Loss: 0.517900696985
    Training Accuracy: 94.3183333333
    Epoch: 719
    Loss: 0.554907956351
    Training Accuracy: 94.4883333333
    Epoch: 720
    Loss: 0.468016530425
    Training Accuracy: 94.4516666667
    Epoch: 721
    Loss: 0.532983763852
    Training Accuracy: 94.38
    Epoch: 722
    Loss: 0.514753520235
    Training Accuracy: 94.4116666667
    Epoch: 723
    Loss: 0.510222121191
    Training Accuracy: 94.5116666667
    Epoch: 724
    Loss: 0.537568039058
    Training Accuracy: 94.3783333333
    Epoch: 725
    Loss: 0.541958225174
    Training Accuracy: 94.5016666667
    Epoch: 726
    Loss: 0.523391831363
    Training Accuracy: 94.4566666667
    Epoch: 727
    Loss: 0.529265176654
    Training Accuracy: 94.6
    Epoch: 728
    Loss: 0.519147530795
    Training Accuracy: 94.7066666667
    Epoch: 729
    Loss: 0.508808887079
    Training Accuracy: 94.5533333333
    Epoch: 730
    Loss: 0.576654383885
    Training Accuracy: 94.5083333333
    Epoch: 731
    Loss: 0.531489969287
    Training Accuracy: 94.61
    Epoch: 732
    Loss: 0.526881633098
    Training Accuracy: 94.5166666667
    Epoch: 733
    Loss: 0.464179545993
    Training Accuracy: 94.6066666667
    Epoch: 734
    Loss: 0.419686991885
    Training Accuracy: 94.67
    Epoch: 735
    Loss: 0.533158158744
    Training Accuracy: 94.6483333333
    Epoch: 736
    Loss: 0.489361745996
    Training Accuracy: 94.48
    Epoch: 737
    Loss: 0.542839105829
    Training Accuracy: 94.55
    Epoch: 738
    Loss: 0.522296410496
    Training Accuracy: 94.5083333333
    Epoch: 739
    Loss: 0.572880541672
    Training Accuracy: 94.5566666667
    Epoch: 740
    Loss: 0.499421821049
    Training Accuracy: 94.4816666667
    Epoch: 741
    Loss: 0.585075745299
    Training Accuracy: 94.6016666667
    Epoch: 742
    Loss: 0.551525879229
    Training Accuracy: 94.715
    Epoch: 743
    Loss: 0.602012507357
    Training Accuracy: 94.6433333333
    Epoch: 744
    Loss: 0.485757815326
    Training Accuracy: 94.6383333333
    Epoch: 745
    Loss: 0.504940939241
    Training Accuracy: 94.6716666667
    Epoch: 746
    Loss: 0.593226657548
    Training Accuracy: 94.6483333333
    Epoch: 747
    Loss: 0.555278698382
    Training Accuracy: 94.6966666667
    Epoch: 748
    Loss: 0.49880182469
    Training Accuracy: 94.53
    Epoch: 749
    Loss: 0.475496703601
    Training Accuracy: 94.5883333333
    Epoch: 750
    Loss: 0.504632551303
    Training Accuracy: 94.505
    Epoch: 751
    Loss: 0.508696368631
    Training Accuracy: 94.6433333333
    Epoch: 752
    Loss: 0.473155485831
    Training Accuracy: 94.5833333333
    Epoch: 753
    Loss: 0.499443459065
    Training Accuracy: 94.5833333333
    Epoch: 754
    Loss: 0.454327226872
    Training Accuracy: 94.6433333333
    Epoch: 755
    Loss: 0.472124205108
    Training Accuracy: 94.675
    Epoch: 756
    Loss: 0.570224260507
    Training Accuracy: 94.595
    Epoch: 757
    Loss: 0.513378316542
    Training Accuracy: 94.7233333333
    Epoch: 758
    Loss: 0.472999083716
    Training Accuracy: 94.66
    Epoch: 759
    Loss: 0.537280018894
    Training Accuracy: 94.8666666667
    Epoch: 760
    Loss: 0.486818311195
    Training Accuracy: 94.765
    Epoch: 761
    Loss: 0.471712923128
    Training Accuracy: 94.8383333333
    Epoch: 762
    Loss: 0.464456765462
    Training Accuracy: 94.69
    Epoch: 763
    Loss: 0.43189372062
    Training Accuracy: 94.8683333333
    Epoch: 764
    Loss: 0.519900282188
    Training Accuracy: 94.8216666667
    Epoch: 765
    Loss: 0.498897771804
    Training Accuracy: 94.7933333333
    Epoch: 766
    Loss: 0.526598564415
    Training Accuracy: 94.7783333333
    Epoch: 767
    Loss: 0.550354282018
    Training Accuracy: 94.7866666667
    Epoch: 768
    Loss: 0.425185811399
    Training Accuracy: 94.8
    Epoch: 769
    Loss: 0.512463311258
    Training Accuracy: 94.7516666667
    Epoch: 770
    Loss: 0.581648507835
    Training Accuracy: 94.795
    Epoch: 771
    Loss: 0.543685083797
    Training Accuracy: 94.915
    Epoch: 772
    Loss: 0.492526825148
    Training Accuracy: 94.9
    Epoch: 773
    Loss: 0.521398336974
    Training Accuracy: 94.7533333333
    Epoch: 774
    Loss: 0.471900317779
    Training Accuracy: 94.81
    Epoch: 775
    Loss: 0.590274478976
    Training Accuracy: 94.7983333333
    Epoch: 776
    Loss: 0.476504433963
    Training Accuracy: 94.5866666667
    Epoch: 777
    Loss: 0.555537384841
    Training Accuracy: 94.87
    Epoch: 778
    Loss: 0.454571177584
    Training Accuracy: 94.87
    Epoch: 779
    Loss: 0.443594684854
    Training Accuracy: 94.8833333333
    Epoch: 780
    Loss: 0.445601963306
    Training Accuracy: 94.8383333333
    Epoch: 781
    Loss: 0.522614823001
    Training Accuracy: 94.8833333333
    Epoch: 782
    Loss: 0.487523272244
    Training Accuracy: 94.765
    Epoch: 783
    Loss: 0.51783402173
    Training Accuracy: 94.8133333333
    Epoch: 784
    Loss: 0.574516758945
    Training Accuracy: 94.7983333333
    Epoch: 785
    Loss: 0.500851547639
    Training Accuracy: 94.8266666667
    Epoch: 786
    Loss: 0.501428747976
    Training Accuracy: 94.8266666667
    Epoch: 787
    Loss: 0.547706336151
    Training Accuracy: 94.91
    Epoch: 788
    Loss: 0.398076098855
    Training Accuracy: 94.8383333333
    Epoch: 789
    Loss: 0.527637905007
    Training Accuracy: 94.8383333333
    Epoch: 790
    Loss: 0.475734653418
    Training Accuracy: 94.86
    Epoch: 791
    Loss: 0.453699102355
    Training Accuracy: 94.8733333333
    Epoch: 792
    Loss: 0.445915532785
    Training Accuracy: 94.8283333333
    Epoch: 793
    Loss: 0.526510414625
    Training Accuracy: 94.7833333333
    Epoch: 794
    Loss: 0.493132938482
    Training Accuracy: 94.9016666667
    Epoch: 795
    Loss: 0.468776236361
    Training Accuracy: 94.9966666667
    Epoch: 796
    Loss: 0.481117723706
    Training Accuracy: 95.0266666667
    Epoch: 797
    Loss: 0.44386230811
    Training Accuracy: 94.9816666667
    Epoch: 798
    Loss: 0.492819187968
    Training Accuracy: 94.9366666667
    Epoch: 799
    Loss: 0.465681654932
    Training Accuracy: 94.9866666667
    Epoch: 800
    Loss: 0.46132142761
    Training Accuracy: 95.0516666667
    Epoch: 801
    Loss: 0.503806778251
    Training Accuracy: 95.0516666667
    Epoch: 802
    Loss: 0.42504072633
    Training Accuracy: 95.12
    Epoch: 803
    Loss: 0.508097936082
    Training Accuracy: 95.0366666667
    Epoch: 804
    Loss: 0.475371735685
    Training Accuracy: 94.96
    Epoch: 805
    Loss: 0.569458911589
    Training Accuracy: 95.0716666667
    Epoch: 806
    Loss: 0.470793530685
    Training Accuracy: 95.0883333333
    Epoch: 807
    Loss: 0.578981428923
    Training Accuracy: 95.07
    Epoch: 808
    Loss: 0.447773717169
    Training Accuracy: 95.005
    Epoch: 809
    Loss: 0.527240125439
    Training Accuracy: 95.1
    Epoch: 810
    Loss: 0.519854558802
    Training Accuracy: 94.9833333333
    Epoch: 811
    Loss: 0.429247426246
    Training Accuracy: 95.135
    Epoch: 812
    Loss: 0.463083511506
    Training Accuracy: 95.09
    Epoch: 813
    Loss: 0.401995490046
    Training Accuracy: 94.99
    Epoch: 814
    Loss: 0.464881177007
    Training Accuracy: 95.07
    Epoch: 815
    Loss: 0.497197259571
    Training Accuracy: 94.9966666667
    Epoch: 816
    Loss: 0.479568749545
    Training Accuracy: 95.0266666667
    Epoch: 817
    Loss: 0.446763228081
    Training Accuracy: 94.94
    Epoch: 818
    Loss: 0.431733632257
    Training Accuracy: 95.0783333333
    Epoch: 819
    Loss: 0.417510040509
    Training Accuracy: 95.1383333333
    Epoch: 820
    Loss: 0.414031554979
    Training Accuracy: 95.0416666667
    Epoch: 821
    Loss: 0.428576409783
    Training Accuracy: 95.1233333333
    Epoch: 822
    Loss: 0.391293118449
    Training Accuracy: 95.025
    Epoch: 823
    Loss: 0.428097971356
    Training Accuracy: 95.09
    Epoch: 824
    Loss: 0.484726068198
    Training Accuracy: 95.0516666667
    Epoch: 825
    Loss: 0.455182966653
    Training Accuracy: 95.1833333333
    Epoch: 826
    Loss: 0.458151237914
    Training Accuracy: 95.0516666667
    Epoch: 827
    Loss: 0.465830969526
    Training Accuracy: 95.1283333333
    Epoch: 828
    Loss: 0.451776870172
    Training Accuracy: 95.1983333333
    Epoch: 829
    Loss: 0.473900148263
    Training Accuracy: 95.1166666667
    Epoch: 830
    Loss: 0.478561665835
    Training Accuracy: 95.1916666667
    Epoch: 831
    Loss: 0.43623012076
    Training Accuracy: 95.1216666667
    Epoch: 832
    Loss: 0.403473118695
    Training Accuracy: 95.23
    Epoch: 833
    Loss: 0.483895610468
    Training Accuracy: 95.19
    Epoch: 834
    Loss: 0.401932530375
    Training Accuracy: 94.96
    Epoch: 835
    Loss: 0.424070315961
    Training Accuracy: 95.0966666667
    Epoch: 836
    Loss: 0.463830529353
    Training Accuracy: 95.135
    Epoch: 837
    Loss: 0.553220702276
    Training Accuracy: 95.135
    Epoch: 838
    Loss: 0.517617732864
    Training Accuracy: 95.2166666667
    Epoch: 839
    Loss: 0.473995403839
    Training Accuracy: 95.125
    Epoch: 840
    Loss: 0.419502085998
    Training Accuracy: 95.16
    Epoch: 841
    Loss: 0.452007982714
    Training Accuracy: 95.1416666667
    Epoch: 842
    Loss: 0.491759642253
    Training Accuracy: 95.2533333333
    Epoch: 843
    Loss: 0.395022063737
    Training Accuracy: 95.1833333333
    Epoch: 844
    Loss: 0.406016741454
    Training Accuracy: 95.195
    Epoch: 845
    Loss: 0.418988180644
    Training Accuracy: 95.2283333333
    Epoch: 846
    Loss: 0.444924424489
    Training Accuracy: 95.2466666667
    Epoch: 847
    Loss: 0.4775719854
    Training Accuracy: 95.27
    Epoch: 848
    Loss: 0.390574857516
    Training Accuracy: 95.2416666667
    Epoch: 849
    Loss: 0.435304890398
    Training Accuracy: 95.3216666667
    Epoch: 850
    Loss: 0.374051351761
    Training Accuracy: 95.36
    Epoch: 851
    Loss: 0.43875946608
    Training Accuracy: 95.33
    Epoch: 852
    Loss: 0.487256055386
    Training Accuracy: 95.285
    Epoch: 853
    Loss: 0.428420227456
    Training Accuracy: 95.1766666667
    Epoch: 854
    Loss: 0.453915995652
    Training Accuracy: 95.23
    Epoch: 855
    Loss: 0.415531667421
    Training Accuracy: 95.3366666667
    Epoch: 856
    Loss: 0.356846419594
    Training Accuracy: 95.27
    Epoch: 857
    Loss: 0.479096564884
    Training Accuracy: 95.21
    Epoch: 858
    Loss: 0.483070392811
    Training Accuracy: 95.2883333333
    Epoch: 859
    Loss: 0.403545072063
    Training Accuracy: 95.2416666667
    Epoch: 860
    Loss: 0.43949498613
    Training Accuracy: 95.2366666667
    Epoch: 861
    Loss: 0.39439330989
    Training Accuracy: 95.3
    Epoch: 862
    Loss: 0.509060508316
    Training Accuracy: 95.2733333333
    Epoch: 863
    Loss: 0.450834884959
    Training Accuracy: 95.2666666667
    Epoch: 864
    Loss: 0.360592355709
    Training Accuracy: 95.2316666667
    Epoch: 865
    Loss: 0.381671770468
    Training Accuracy: 95.3383333333
    Epoch: 866
    Loss: 0.383942622622
    Training Accuracy: 95.28
    Epoch: 867
    Loss: 0.427322070451
    Training Accuracy: 95.2483333333
    Epoch: 868
    Loss: 0.417490094889
    Training Accuracy: 95.2983333333
    Epoch: 869
    Loss: 0.402063531411
    Training Accuracy: 95.285
    Epoch: 870
    Loss: 0.469501089613
    Training Accuracy: 95.245
    Epoch: 871
    Loss: 0.446643690096
    Training Accuracy: 95.285
    Epoch: 872
    Loss: 0.458214794282
    Training Accuracy: 95.3166666667
    Epoch: 873
    Loss: 0.487960375438
    Training Accuracy: 95.2566666667
    Epoch: 874
    Loss: 0.460323031429
    Training Accuracy: 95.3533333333
    Epoch: 875
    Loss: 0.49147079906
    Training Accuracy: 95.315
    Epoch: 876
    Loss: 0.522180733464
    Training Accuracy: 95.3316666667
    Epoch: 877
    Loss: 0.43044982089
    Training Accuracy: 95.3416666667
    Epoch: 878
    Loss: 0.398506253291
    Training Accuracy: 95.3733333333
    Epoch: 879
    Loss: 0.42630680373
    Training Accuracy: 95.3166666667
    Epoch: 880
    Loss: 0.392821096413
    Training Accuracy: 95.41
    Epoch: 881
    Loss: 0.44515757378
    Training Accuracy: 95.4783333333
    Epoch: 882
    Loss: 0.412937872552
    Training Accuracy: 95.4566666667
    Epoch: 883
    Loss: 0.452032924159
    Training Accuracy: 95.4116666667
    Epoch: 884
    Loss: 0.456539284097
    Training Accuracy: 95.4216666667
    Epoch: 885
    Loss: 0.410461947365
    Training Accuracy: 95.3783333333
    Epoch: 886
    Loss: 0.40208001467
    Training Accuracy: 95.455
    Epoch: 887
    Loss: 0.418400930089
    Training Accuracy: 95.3783333333
    Epoch: 888
    Loss: 0.398961088857
    Training Accuracy: 95.3683333333
    Epoch: 889
    Loss: 0.456471675065
    Training Accuracy: 95.47
    Epoch: 890
    Loss: 0.432595376088
    Training Accuracy: 95.51
    Epoch: 891
    Loss: 0.373837224958
    Training Accuracy: 95.4933333333
    Epoch: 892
    Loss: 0.440242779177
    Training Accuracy: 95.5366666667
    Epoch: 893
    Loss: 0.495021588597
    Training Accuracy: 95.3666666667
    Epoch: 894
    Loss: 0.383287650561
    Training Accuracy: 95.545
    Epoch: 895
    Loss: 0.379282068635
    Training Accuracy: 95.4733333333
    Epoch: 896
    Loss: 0.409773509056
    Training Accuracy: 95.4966666667
    Epoch: 897
    Loss: 0.40451803187
    Training Accuracy: 95.4716666667
    Epoch: 898
    Loss: 0.40794678116
    Training Accuracy: 95.4116666667
    Epoch: 899
    Loss: 0.370939355485
    Training Accuracy: 95.4783333333
    Epoch: 900
    Loss: 0.349781765144
    Training Accuracy: 95.4216666667
    Epoch: 901
    Loss: 0.44314689112
    Training Accuracy: 95.4366666667
    Epoch: 902
    Loss: 0.366372838736
    Training Accuracy: 95.46
    Epoch: 903
    Loss: 0.426934609946
    Training Accuracy: 95.47
    Epoch: 904
    Loss: 0.397136666745
    Training Accuracy: 95.415
    Epoch: 905
    Loss: 0.350556999117
    Training Accuracy: 95.4866666667
    Epoch: 906
    Loss: 0.424807944926
    Training Accuracy: 95.455
    Epoch: 907
    Loss: 0.34231188269
    Training Accuracy: 95.4333333333
    Epoch: 908
    Loss: 0.418757417893
    Training Accuracy: 95.4183333333
    Epoch: 909
    Loss: 0.384836292067
    Training Accuracy: 95.435
    Epoch: 910
    Loss: 0.398341800012
    Training Accuracy: 95.4666666667
    Epoch: 911
    Loss: 0.421238347015
    Training Accuracy: 95.425
    Epoch: 912
    Loss: 0.337417104175
    Training Accuracy: 95.4666666667
    Epoch: 913
    Loss: 0.387119605228
    Training Accuracy: 95.4866666667
    Epoch: 914
    Loss: 0.388110478187
    Training Accuracy: 95.4633333333
    Epoch: 915
    Loss: 0.419657200935
    Training Accuracy: 95.4566666667
    Epoch: 916
    Loss: 0.369750048773
    Training Accuracy: 95.5766666667
    Epoch: 917
    Loss: 0.370978274535
    Training Accuracy: 95.545
    Epoch: 918
    Loss: 0.435109754083
    Training Accuracy: 95.5633333333
    Epoch: 919
    Loss: 0.369341990935
    Training Accuracy: 95.5216666667
    Epoch: 920
    Loss: 0.331792611938
    Training Accuracy: 95.505
    Epoch: 921
    Loss: 0.442581001985
    Training Accuracy: 95.5833333333
    Epoch: 922
    Loss: 0.492920657657
    Training Accuracy: 95.5
    Epoch: 923
    Loss: 0.330385697417
    Training Accuracy: 95.5316666667
    Epoch: 924
    Loss: 0.438986959312
    Training Accuracy: 95.585
    Epoch: 925
    Loss: 0.417799683475
    Training Accuracy: 95.4916666667
    Epoch: 926
    Loss: 0.397760477904
    Training Accuracy: 95.5616666667
    Epoch: 927
    Loss: 0.436653460799
    Training Accuracy: 95.5316666667
    Epoch: 928
    Loss: 0.355497615499
    Training Accuracy: 95.5133333333
    Epoch: 929
    Loss: 0.415408382469
    Training Accuracy: 95.5233333333
    Epoch: 930
    Loss: 0.405731019191
    Training Accuracy: 95.5333333333
    Epoch: 931
    Loss: 0.365376422473
    Training Accuracy: 95.5366666667
    Epoch: 932
    Loss: 0.400619269527
    Training Accuracy: 95.6183333333
    Epoch: 933
    Loss: 0.370250482793
    Training Accuracy: 95.5666666667
    Epoch: 934
    Loss: 0.488481667332
    Training Accuracy: 95.5616666667
    Epoch: 935
    Loss: 0.423630363948
    Training Accuracy: 95.59
    Epoch: 936
    Loss: 0.363565862329
    Training Accuracy: 95.595
    Epoch: 937
    Loss: 0.331352158146
    Training Accuracy: 95.6616666667
    Epoch: 938
    Loss: 0.36582954261
    Training Accuracy: 95.6483333333
    Epoch: 939
    Loss: 0.415610479702
    Training Accuracy: 95.61
    Epoch: 940
    Loss: 0.321129983366
    Training Accuracy: 95.6766666667
    Epoch: 941
    Loss: 0.347039576207
    Training Accuracy: 95.6016666667
    Epoch: 942
    Loss: 0.410919738249
    Training Accuracy: 95.5816666667
    Epoch: 943
    Loss: 0.380961700321
    Training Accuracy: 95.6266666667
    Epoch: 944
    Loss: 0.43684000602
    Training Accuracy: 95.6633333333
    Epoch: 945
    Loss: 0.378905409062
    Training Accuracy: 95.65
    Epoch: 946
    Loss: 0.433080557199
    Training Accuracy: 95.5816666667
    Epoch: 947
    Loss: 0.423844553701
    Training Accuracy: 95.6133333333
    Epoch: 948
    Loss: 0.380776762466
    Training Accuracy: 95.61
    Epoch: 949
    Loss: 0.429678756067
    Training Accuracy: 95.6033333333
    Epoch: 950
    Loss: 0.345779187512
    Training Accuracy: 95.6333333333
    Epoch: 951
    Loss: 0.328943168864
    Training Accuracy: 95.5633333333
    Epoch: 952
    Loss: 0.369249168066
    Training Accuracy: 95.565
    Epoch: 953
    Loss: 0.461447984568
    Training Accuracy: 95.635
    Epoch: 954
    Loss: 0.393544355395
    Training Accuracy: 95.5483333333
    Epoch: 955
    Loss: 0.365934792655
    Training Accuracy: 95.6333333333
    Epoch: 956
    Loss: 0.372681345274
    Training Accuracy: 95.6116666667
    Epoch: 957
    Loss: 0.414633132859
    Training Accuracy: 95.6116666667
    Epoch: 958
    Loss: 0.382289926814
    Training Accuracy: 95.6666666667
    Epoch: 959
    Loss: 0.496062298685
    Training Accuracy: 95.615
    Epoch: 960
    Loss: 0.411463934564
    Training Accuracy: 95.6033333333
    Epoch: 961
    Loss: 0.427734538664
    Training Accuracy: 95.61
    Epoch: 962
    Loss: 0.425394557367
    Training Accuracy: 95.6016666667
    Epoch: 963
    Loss: 0.422828157616
    Training Accuracy: 95.6283333333
    Epoch: 964
    Loss: 0.428843273821
    Training Accuracy: 95.6183333333
    Epoch: 965
    Loss: 0.383504390425
    Training Accuracy: 95.5716666667
    Epoch: 966
    Loss: 0.44359465888
    Training Accuracy: 95.65
    Epoch: 967
    Loss: 0.440071921547
    Training Accuracy: 95.6016666667
    Epoch: 968
    Loss: 0.393153691645
    Training Accuracy: 95.6266666667
    Epoch: 969
    Loss: 0.470657314312
    Training Accuracy: 95.645
    Epoch: 970
    Loss: 0.382381322464
    Training Accuracy: 95.6133333333
    Epoch: 971
    Loss: 0.395752807398
    Training Accuracy: 95.6566666667
    Epoch: 972
    Loss: 0.344586140208
    Training Accuracy: 95.6116666667
    Epoch: 973
    Loss: 0.418142635505
    Training Accuracy: 95.6666666667
    Epoch: 974
    Loss: 0.457688490535
    Training Accuracy: 95.635
    Epoch: 975
    Loss: 0.329026512238
    Training Accuracy: 95.6883333333
    Epoch: 976
    Loss: 0.396385566106
    Training Accuracy: 95.64
    Epoch: 977
    Loss: 0.378201332275
    Training Accuracy: 95.6466666667
    Epoch: 978
    Loss: 0.37831246607
    Training Accuracy: 95.6666666667
    Epoch: 979
    Loss: 0.415932450298
    Training Accuracy: 95.675
    Epoch: 980
    Loss: 0.347570708057
    Training Accuracy: 95.6433333333
    Epoch: 981
    Loss: 0.397802756122
    Training Accuracy: 95.6983333333
    Epoch: 982
    Loss: 0.375693029831
    Training Accuracy: 95.655
    Epoch: 983
    Loss: 0.385562114062
    Training Accuracy: 95.6933333333
    Epoch: 984
    Loss: 0.459569397165
    Training Accuracy: 95.7333333333
    Epoch: 985
    Loss: 0.353694086359
    Training Accuracy: 95.755
    Epoch: 986
    Loss: 0.416926091482
    Training Accuracy: 95.71
    Epoch: 987
    Loss: 0.344336671566
    Training Accuracy: 95.7183333333
    Epoch: 988
    Loss: 0.430041189469
    Training Accuracy: 95.6666666667
    Epoch: 989
    Loss: 0.317049780866
    Training Accuracy: 95.675
    Epoch: 990
    Loss: 0.451602027611
    Training Accuracy: 95.62
    Epoch: 991
    Loss: 0.362296239352
    Training Accuracy: 95.6916666667
    Epoch: 992
    Loss: 0.39487265226
    Training Accuracy: 95.6666666667
    Epoch: 993
    Loss: 0.413807122698
    Training Accuracy: 95.73
    Epoch: 994
    Loss: 0.380897782969
    Training Accuracy: 95.6533333333
    Epoch: 995
    Loss: 0.353138018089
    Training Accuracy: 95.65
    Epoch: 996
    Loss: 0.365107583536
    Training Accuracy: 95.6616666667
    Epoch: 997
    Loss: 0.346923960997
    Training Accuracy: 95.68
    Epoch: 998
    Loss: 0.383173167342
    Training Accuracy: 95.675
    Epoch: 999
    Loss: 0.379563914882
    Training Accuracy: 95.625
    Epoch: 1000
    Loss: 0.397607041169
    Training Accuracy: 95.6566666667





    <NeuralNetwork.NeuralNetwork at 0x10a1f8b90>




```python

```


```python
y_train_pred = nn.predict(X_train)
print y_train_pred
```

    [5 0 4 ..., 5 6 8]



```python
print y_train.shape[0]
print y_train_pred.shape[0]
diffs = y_train_pred  - y_train
count = 0.
for i in range(y_train.shape[0]):
    if diffs[i] != 0:
        count = count + 1
print 100 - count*100/y_train.shape[0]
    
```

    60000
    60000
    95.795



```python
y_test_pred = nn.predict(X_test)
print y_test_pred
```

    [7 2 1 ..., 4 5 6]



```python
print y_test_pred.shape[0]
print y_test.shape[0]
diffs = y_test_pred - y_test
mistakes = []
count = 0.
for i in range(y_test.shape[0]):
    if diffs[i] != 0:
        count = count + 1
        mistakes.append({"actual": y_test[i], "predicted": y_test_pred[i]})
print 100 - count*100/y_test.shape[0]
# initialize a data structure where each item will keep track of what mispredictions there were
mistake_categories = [(i, {k:0 for k in range(10) if k != i}) for i in range(10)]
for mistake in mistakes:
    mistake_categories[mistake['actual']][1][mistake['predicted']]+=1

for tup in mistake_categories:
    for k, v in tup[1].items():
        if v == 0: del tup[1][k] # remove keys where no mistakes were found 

for i in range(len(mistake_categories)): print mistake_categories[i]
# tells us our mispredictions - ex: for images with label 2, we mispredicted a 0 twelve times, and mispredicted it as an 8 twenty times. 
# print [mistakes[i] for i in range(len(mistakes)) if mistakes[i]['actual']==4]

```

    10000
    10000
    95.73
    (0, {3: 1, 5: 2, 6: 4, 7: 2, 8: 2})
    (1, {2: 3, 3: 2, 6: 3, 8: 5, 9: 1})
    (2, {0: 12, 3: 8, 4: 4, 6: 9, 7: 7, 8: 20, 9: 2})
    (3, {0: 1, 2: 15, 5: 10, 6: 1, 7: 11, 8: 13, 9: 2})
    (4, {0: 1, 2: 3, 6: 12, 8: 5, 9: 31})
    (5, {0: 4, 1: 1, 2: 1, 3: 11, 4: 1, 6: 13, 7: 1, 8: 11, 9: 3})
    (6, {0: 9, 1: 3, 2: 1, 4: 4, 5: 6, 8: 5})
    (7, {0: 3, 1: 4, 2: 20, 3: 4, 4: 3, 8: 5, 9: 26})
    (8, {0: 5, 1: 2, 2: 2, 3: 6, 4: 5, 5: 4, 6: 7, 7: 5, 9: 7})
    (9, {0: 7, 1: 6, 2: 1, 3: 11, 4: 14, 5: 1, 6: 1, 7: 3, 8: 9})



```python

```


```python

```
