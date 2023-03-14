import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt 
data = {
    'emp_no' : [101,102,103,104,105,106,107,108,109,110],
    'expirence' : [4,2,6,11,8,9,20,15,12,1],
    'salery' : [8,6,10,15,12,13,25,18,16,4]
    }
df = pd.DataFrame(data)
df.plot(x='expirence',y='salery',kind = 'scatter')
plt.show()
print(data)
print(df)