# Sign_Spotting

|                  Setting                  | Precious |  Recall  | F1 Score |
| :---------------------------------------: | :------: | :------: | :------: |
|             I3D (duration=8)              |   0.51   |   0.29   |   0.47   |
|             I3D (duration=16)             |   0.34   |   0.29   |   0.31   |
|             I3D (duration=4)              |   0.36   |   0.24   |   0.28   |
|                  TMODEL                   |   0.51   |   0.38   |   0.44   |
|            TMODEL (only conv)             |   0.53   |   0.36   |   0.43   |
|         TMODEL (share classifier)         | **0.55** | **0.43** | **0.48** |
|           TMODEL (weight norm)            |   0.53   |   0.39   |   0.45   |
|      TMODEL (weight norm+feat norm)       |   0.49   |   0.36   |   0.42   |
| TMODEL (share classifier+label smoothing) |   0.55   |   0.39   |   0.46   |

