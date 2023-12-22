# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:14.419229Z","iopub.execute_input":"2023-11-26T08:23:14.419607Z","iopub.status.idle":"2023-11-26T08:23:14.429137Z","shell.execute_reply.started":"2023-11-26T08:23:14.419581Z","shell.execute_reply":"2023-11-26T08:23:14.428203Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
# Input data files are available in the read-only "../input/" directoryfrom sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve,log_loss, f1_score
from sklearn.model_selection import KFold

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import catboost as catboost
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # Load Data

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:14.857484Z","iopub.execute_input":"2023-11-26T08:23:14.857825Z","iopub.status.idle":"2023-11-26T08:23:14.890833Z","shell.execute_reply.started":"2023-11-26T08:23:14.857798Z","shell.execute_reply":"2023-11-26T08:23:14.889910Z"}}
train = pd.read_csv('/kaggle/input/playground-series-s3e22/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e22/test.csv')

sample_submission = pd.read_csv('/kaggle/input/playground-series-s3e22/sample_submission.csv')
origin = pd.read_csv('/kaggle/input/horse-survival-dataset/horse.csv')

#Drop ID
train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

train_total = pd.concat([train,origin], ignore_index=True)

total = pd.concat([train_total,test],ignore_index=True)

print("Shape Train Data: ", train.shape)
print("Shape Test Data: ", test.shape)
print("Shape Total Data: ", total.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:14.892707Z","iopub.execute_input":"2023-11-26T08:23:14.892965Z","iopub.status.idle":"2023-11-26T08:23:14.897476Z","shell.execute_reply.started":"2023-11-26T08:23:14.892943Z","shell.execute_reply":"2023-11-26T08:23:14.896431Z"}}
df_train = train.copy()
df_test = test.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:15.309195Z","iopub.execute_input":"2023-11-26T08:23:15.309514Z","iopub.status.idle":"2023-11-26T08:23:15.329849Z","shell.execute_reply.started":"2023-11-26T08:23:15.309489Z","shell.execute_reply":"2023-11-26T08:23:15.329234Z"}}
df_train.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:16.417091Z","iopub.execute_input":"2023-11-26T08:23:16.418135Z","iopub.status.idle":"2023-11-26T08:23:16.426564Z","shell.execute_reply.started":"2023-11-26T08:23:16.418092Z","shell.execute_reply":"2023-11-26T08:23:16.425466Z"}}
df_train['age'].value_counts()
df_test['age'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:16.428251Z","iopub.execute_input":"2023-11-26T08:23:16.429059Z","iopub.status.idle":"2023-11-26T08:23:16.436566Z","shell.execute_reply.started":"2023-11-26T08:23:16.429026Z","shell.execute_reply":"2023-11-26T08:23:16.435514Z"}}
def summary(df):
    data = pd.DataFrame(index=df.columns)
    data['dtypes'] = df.dtypes
    data['count'] = df.count()
    data['#unique'] = df.nunique()
    data['#missing'] = df.isna().sum()
    data['%missing'] = df.isna().sum()/len(df)*100
    data = pd.concat([data,df.describe().T.drop('count',axis=1)],axis=1)
    return data

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:16.893284Z","iopub.execute_input":"2023-11-26T08:23:16.893618Z","iopub.status.idle":"2023-11-26T08:23:16.958372Z","shell.execute_reply.started":"2023-11-26T08:23:16.893592Z","shell.execute_reply":"2023-11-26T08:23:16.957431Z"}}
summary(train).style.background_gradient(cmap='YlGnBu')

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:17.531621Z","iopub.execute_input":"2023-11-26T08:23:17.531973Z","iopub.status.idle":"2023-11-26T08:23:17.594144Z","shell.execute_reply.started":"2023-11-26T08:23:17.531947Z","shell.execute_reply":"2023-11-26T08:23:17.593167Z"}}
#summary test dataset
summary(test).style.background_gradient(cmap='YlOrBr')

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:17.924240Z","iopub.execute_input":"2023-11-26T08:23:17.924564Z","iopub.status.idle":"2023-11-26T08:23:18.117381Z","shell.execute_reply.started":"2023-11-26T08:23:17.924540Z","shell.execute_reply":"2023-11-26T08:23:18.116579Z"}}
sns.countplot(data=train,x='outcome')
plt.title("Categorical count of Outcome")

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:18.899306Z","iopub.execute_input":"2023-11-26T08:23:18.899666Z","iopub.status.idle":"2023-11-26T08:23:18.904214Z","shell.execute_reply.started":"2023-11-26T08:23:18.899638Z","shell.execute_reply":"2023-11-26T08:23:18.903555Z"}}
categorical_cols = ['surgery', 'age', 'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time',
                   'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces',
                   'abdomen', 'abdomo_appearance', 'surgical_lesion', 'cp_data']

num_cols = ['hospital_number', 'rectal_temp', 'pulse', 'respiratory_rate', 'nasogastric_reflux_ph', 'packed_cell_volume', 'total_protein',
           'abdomo_protein', 'lesion_1', 'lesion_2', 'lesion_3']

# %% [code]


# %% [code]


# %% [raw]
# # Data Exploration
# 

# %% [markdown]
# ****Count Plot Categorical Columns****

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T06:37:46.040480Z","iopub.execute_input":"2023-11-26T06:37:46.041231Z","iopub.status.idle":"2023-11-26T06:37:46.053864Z","shell.execute_reply.started":"2023-11-26T06:37:46.041185Z","shell.execute_reply":"2023-11-26T06:37:46.052482Z"}}
def plot_count(df,columns,n_cols,hue):
    '''
    Function to generate countplot
    df: total data
    columns: category variables
    n_cols: num of cols
    '''
    
    n_rows = (len(columns)-1)//n_cols+1
    fig,ax = plt.subplots(n_rows,n_cols,figsize=(17,4*n_rows))
    ax = ax.flatten()
    
    for i, column in enumerate(columns):
        sns.countplot(data=df,x=column,ax=ax[i],hue=hue)
        ax[i].set_title(f'{column} Counts', fontsize=18)
        ax[i].set_xlabel(None, fontsize=16)
        ax[i].set_ylabel(None, fontsize=16)
        ax[i].tick_params(axis='x', rotation=10)
        for p in ax[i].patches:
            value = int(p.get_height())
            ax[i].annotate(f'{value:.0f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                           ha='center', va='bottom', fontsize=9)
        ylim_top = ax[i].get_ylim()[1]
        ax[i].set_ylim(top=ylim_top * 1.1)
    for i in range(len(columns), len(ax)):
        ax[i].axis('off')
    plt.tight_layout()
    plt.savefig("output.jpg")

    
# plot_count(train,categorical_cols,3,'outcome')

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T06:37:46.204042Z","iopub.execute_input":"2023-11-26T06:37:46.204815Z","iopub.status.idle":"2023-11-26T06:37:46.240426Z","shell.execute_reply.started":"2023-11-26T06:37:46.204761Z","shell.execute_reply":"2023-11-26T06:37:46.238814Z"}}
plot_count(test,categorical_cols,3,None)

# %% [markdown]
# Pair Plot Numerical Features

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T06:37:46.637135Z","iopub.execute_input":"2023-11-26T06:37:46.637932Z","iopub.status.idle":"2023-11-26T06:37:46.671034Z","shell.execute_reply.started":"2023-11-26T06:37:46.637870Z","shell.execute_reply":"2023-11-26T06:37:46.669668Z"}}
def pair_plot(df_train,num_var,target, plotname):
    '''
        Funtion to make a pairplot:
    df_train: total data
    num_var: a list of numeric variable
    target: target variable
    '''
    g = sns.pairplot(data=df_train, x_vars = num_var, y_vars= num_var, hue=target, corner=True)
#     g._legend.set_bbox_to_anchor
    g._legend.set_title(target)
    g._legend.loc = 'upper center'
    g._legend.get_title().set_fontsize(14)
    for item in g._legend.get_texts():
        item.set_fontsize(14)
    plt.suptitle(plotname, ha='center', fontweight='bold', fontsize=25, y=1)
    plt.show()
pair_plot(train, num_cols, 'outcome','Scatter Matrix')

# %% [markdown]
# Not many related numerical features

# %% [markdown]
# **Box Plots**

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T06:37:47.968882Z","iopub.execute_input":"2023-11-26T06:37:47.969273Z","iopub.status.idle":"2023-11-26T06:37:48.014497Z","shell.execute_reply.started":"2023-11-26T06:37:47.969244Z","shell.execute_reply":"2023-11-26T06:37:48.013507Z"}}
df = pd.concat([train[num_cols].assign(Source='Train'),
                test[num_cols].assign(Source='Test')],
              axis=0, ignore_index = True)

fig, axes = plt.subplots(len(num_cols),3,figsize = (16, len(num_cols)*4.2),
                        gridspec_kw={'hspace':0.35,'wspace':0.3, 'width_ratios': [0.8,0.2,0.2]})

for i, col in enumerate(num_cols):
    ax = axes[i,0]
    sns.kdeplot(data=df[[col,'Source']], x = col, hue='Source',
               ax=ax, linewidth=2.1)
    ax.set_title(f"\n{col}", fontsize=9, fontweight='bold')
    ax.grid(visible=True, which = 'both', linestyle = '--', color='lightgrey', linewidth = 0.75);
    ax.set(xlabel = '', ylabel = '');
    
    ax = axes[i,1]
    sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25,saturation = 0.90, linewidth = 0.90, fliersize= 2.25, color = '#037d97',
                ax = ax)
    ax.set(xlabel = '', ylabel = '');
    ax.set_title(f"Train",fontsize = 9, fontweight= 'bold');
    
    ax = axes[i,2]
    sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, fliersize= 2.25,
                saturation = 0.6, linewidth = 0.90, color = '#E4591E',
                ax = ax); 
    ax.set(xlabel = '', ylabel = '');
    ax.set_title(f"Test",fontsize = 9, fontweight= 'bold');

plt.tight_layout();
plt.show();


# %% [markdown]
# HistPlot

# %% [code] {"execution":{"iopub.status.busy":"2023-11-25T14:31:24.743660Z","iopub.execute_input":"2023-11-25T14:31:24.744184Z","iopub.status.idle":"2023-11-25T14:31:35.317177Z","shell.execute_reply.started":"2023-11-25T14:31:24.744144Z","shell.execute_reply":"2023-11-25T14:31:35.315581Z"}}
plt.figure(figsize=(14, len(num_cols) * 2.5))

for idx, column in enumerate(num_cols):
    plt.subplot(len(num_cols), 2, idx*2+1)
    sns.histplot(x=column, hue="outcome", data=train, bins=30, kde=True)
    plt.title(f"{column} Distribution for outcome")
    plt.ylim(0, train[column].value_counts().max() + 10)
    
plt.tight_layout()
plt.savefig("output1.png")


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T06:38:44.617877Z","iopub.execute_input":"2023-11-26T06:38:44.618298Z","iopub.status.idle":"2023-11-26T06:38:45.368678Z","shell.execute_reply.started":"2023-11-26T06:38:44.618264Z","shell.execute_reply":"2023-11-26T06:38:45.367638Z"}}
corr_matrix = train[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='Oranges', fmt='.2f', linewidths=1, square=True, annot_kws={"size": 9} )
plt.title('Correlation Matrix', fontsize=15)
plt.savefig("corr.png")


# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true,"execution":{"iopub.status.busy":"2023-11-25T14:18:55.971072Z","iopub.execute_input":"2023-11-25T14:18:55.971565Z","iopub.status.idle":"2023-11-25T14:19:00.036814Z","shell.execute_reply.started":"2023-11-25T14:18:55.971517Z","shell.execute_reply":"2023-11-25T14:19:00.035222Z"}}
numerical_features = train.select_dtypes(include=['float64']).columns
for i, feature in enumerate(num_cols):
    plt.figure(figsize=(15,18))

    plt.subplot(3,4,i+1)
    sns.boxplot(data=train,x='outcome', y=feature)
    plt.title(f"Distribution of {feature} by Outcome")
    plt.tight_layout()
    plt.show()

    

# box_plot(train, num_cols, 3, 'outcome')

# %% [markdown]
# # Inferences
# 1. Dataset is very small
# 2. Lots of numerical and categorical attributes.
# 3. There are missing values
# 4. Total protein appears to very highly related with the target.
# 5. There is no obvious linear relationship between numerical features.
# 7. The age Counts column contains a large number of adults.
# 8. When cp_data_Counts is Yes, the target variable is euthanized with a small probability.
# 9. Lesion 3 have very less instances.
# 10. Hospital Id is case id which can be dropped 

# %% [markdown]
# Outcome:
# Removing lesion_2 lesion_3 hospital id from train dataset

# %% [markdown]
# **About Missing Data**
# Since most of missing values are in categorical columns, we are going with mode fill strategy.

# %% [markdown]
# Before feature we need to convert out categorical features in numerical data.
# This process is know as encoding.
# There are many types of encoding but 2 are more important
# 1. **Ordinal Encoding**
# 2. **One Hot Encoding**
# 
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:28.953016Z","iopub.execute_input":"2023-11-26T08:23:28.953387Z","iopub.status.idle":"2023-11-26T08:23:28.960744Z","shell.execute_reply.started":"2023-11-26T08:23:28.953362Z","shell.execute_reply":"2023-11-26T08:23:28.959466Z"}}
target='outcome'
train_total[target] = total[target].map({'died':0,'euthanized':1,'lived':2})


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:29.278977Z","iopub.execute_input":"2023-11-26T08:23:29.279325Z","iopub.status.idle":"2023-11-26T08:23:29.291867Z","shell.execute_reply.started":"2023-11-26T08:23:29.279300Z","shell.execute_reply":"2023-11-26T08:23:29.290672Z"}}
def preprocessing(df, le_cols, ohe_cols):
    
    # Label Encoding for binary cols
    le = LabelEncoder()    
    for col in le_cols:
        df[col] = le.fit_transform(df[col])
    
    # OneHot Encoding for category cols
    df = pd.get_dummies(df, columns = ohe_cols)
    
    df["pain"] = df["pain"].replace('slight', 'moderate')
    df["peristalsis"] = df["peristalsis"].replace('distend_small', 'normal')
    df["rectal_exam_feces"] = df["rectal_exam_feces"].replace('serosanguious', 'absent')
    df["nasogastric_reflux"] = df["nasogastric_reflux"].replace('slight', 'none')
        
    df["temp_of_extremities"] = df["temp_of_extremities"].fillna("normal").map({'cold': 0, 'cool': 1, 'normal': 2, 'warm': 3})
    df["peripheral_pulse"] = df["peripheral_pulse"].fillna("normal").map({'absent': 0, 'reduced': 1, 'normal': 2, 'increased': 3})
    df["capillary_refill_time"] = df["capillary_refill_time"].fillna("3").map({'less_3_sec': 0, '3': 1, 'more_3_sec': 2})
    df["pain"] = df["pain"].fillna("depressed").map({'alert': 0, 'depressed': 1, 'moderate': 2, 'mild_pain': 3, 'severe_pain': 4, 'extreme_pain': 5})
    df["peristalsis"] = df["peristalsis"].fillna("hypomotile").map({'hypermotile': 0, 'normal': 1, 'hypomotile': 2, 'absent': 3})
    df["abdominal_distention"] = df["abdominal_distention"].fillna("none").map({'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3})
    df["nasogastric_tube"] = df["nasogastric_tube"].fillna("none").map({'none': 0, 'slight': 1, 'significant': 2})
    df["nasogastric_reflux"] = df["nasogastric_reflux"].fillna("none").map({'less_1_liter': 0, 'none': 1, 'more_1_liter': 2})
    df["rectal_exam_feces"] = df["rectal_exam_feces"].fillna("absent").map({'absent': 0, 'decreased': 1, 'normal': 2, 'increased': 3})
    df["abdomen"] = df["abdomen"].fillna("distend_small").map({'normal': 0, 'other': 1, 'firm': 2,'distend_small': 3, 'distend_large': 4})
    df["abdomo_appearance"] = df["abdomo_appearance"].fillna("serosanguious").map({'clear': 0, 'cloudy': 1, 'serosanguious': 2})
    
    # Imputer 
    missing_values = df.isna().sum()
    missing_values_cols = missing_values[missing_values > 0].index
    df[missing_values_cols]=df[missing_values_cols].fillna(df.mode().iloc[0])
    return df  

def features_engineering(df):
    df['lesion_2'] = df['lesion_2'].apply(lambda x:1 if x>0 else 0)
    data_preprocessed = df.copy()
     
    return data_preprocessed


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:30.572887Z","iopub.execute_input":"2023-11-26T08:23:30.573226Z","iopub.status.idle":"2023-11-26T08:23:30.617553Z","shell.execute_reply.started":"2023-11-26T08:23:30.573200Z","shell.execute_reply":"2023-11-26T08:23:30.616592Z"}}
train_total = preprocessing(train_total, le_cols = ["surgery", "age", "surgical_lesion", "cp_data"], ohe_cols = ["mucous_membrane"])
train_total = features_engineering(train_total)
train_total["abs_rectal_temp"] = (train_total["rectal_temp"] - 37.8).abs()
train_total.drop(columns=["rectal_temp"],inplace=True)


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:32.228961Z","iopub.execute_input":"2023-11-26T08:23:32.229314Z","iopub.status.idle":"2023-11-26T08:23:32.236099Z","shell.execute_reply.started":"2023-11-26T08:23:32.229289Z","shell.execute_reply":"2023-11-26T08:23:32.234762Z"}}
train_total.columns


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:33.268793Z","iopub.execute_input":"2023-11-26T08:23:33.270143Z","iopub.status.idle":"2023-11-26T08:23:33.306525Z","shell.execute_reply.started":"2023-11-26T08:23:33.270097Z","shell.execute_reply":"2023-11-26T08:23:33.305641Z"}}
test = preprocessing(test, le_cols = ["surgery", "age", "surgical_lesion", "cp_data"], ohe_cols = ["mucous_membrane"])
test = features_engineering(test)
test["abs_rectal_temp"] = (test["rectal_temp"] - 37.8).abs()
test.drop(columns=["rectal_temp"],inplace=True)


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:34.224827Z","iopub.execute_input":"2023-11-26T08:23:34.225389Z","iopub.status.idle":"2023-11-26T08:23:34.231522Z","shell.execute_reply.started":"2023-11-26T08:23:34.225358Z","shell.execute_reply":"2023-11-26T08:23:34.230303Z"}}
y = train_total["outcome"]
X = train_total.drop(["outcome","hospital_number","lesion_3","lesion_2"],axis=1)


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:34.904938Z","iopub.execute_input":"2023-11-26T08:23:34.905517Z","iopub.status.idle":"2023-11-26T08:23:34.914178Z","shell.execute_reply.started":"2023-11-26T08:23:34.905486Z","shell.execute_reply":"2023-11-26T08:23:34.912965Z"}}
X["abs_rectal_temp"]

# %% [markdown]
# # **Regularization via Shrinkage**

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:36.244859Z","iopub.execute_input":"2023-11-26T08:23:36.245741Z","iopub.status.idle":"2023-11-26T08:23:36.287426Z","shell.execute_reply.started":"2023-11-26T08:23:36.245697Z","shell.execute_reply":"2023-11-26T08:23:36.286421Z"}}

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
scaler = StandardScaler()
scaler.fit(X)
sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1',solver='liblinear'))
sel_.fit(scaler.transform(X), y)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:36.648740Z","iopub.execute_input":"2023-11-26T08:23:36.649061Z","iopub.status.idle":"2023-11-26T08:23:36.656086Z","shell.execute_reply.started":"2023-11-26T08:23:36.649037Z","shell.execute_reply":"2023-11-26T08:23:36.654828Z"}}
selected_feat = X.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))


# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:39.896720Z","iopub.execute_input":"2023-11-26T08:23:39.897062Z","iopub.status.idle":"2023-11-26T08:23:39.903636Z","shell.execute_reply.started":"2023-11-26T08:23:39.897038Z","shell.execute_reply":"2023-11-26T08:23:39.902807Z"}}
X.columns

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:23:59.375893Z","iopub.execute_input":"2023-11-26T08:23:59.376226Z","iopub.status.idle":"2023-11-26T08:23:59.381270Z","shell.execute_reply.started":"2023-11-26T08:23:59.376191Z","shell.execute_reply":"2023-11-26T08:23:59.379938Z"}}
X_s= X.copy()
y_s = y.copy()


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:30:50.994421Z","iopub.execute_input":"2023-11-26T08:30:50.994733Z","iopub.status.idle":"2023-11-26T08:30:51.010543Z","shell.execute_reply.started":"2023-11-26T08:30:50.994709Z","shell.execute_reply":"2023-11-26T08:30:51.009696Z"}}
import statsmodels.api as sm

# X is your feature matrix, y is the target variable
# Step 1: Fit the model with all features
est = sm.OLS(y_s.astype(float), X_s.astype(float)).fit()

# Step 2-7: Backward Elimination
while True:
    # Step 2: Get p-values for each feature
    p_values = model.pvalues[1:]

    # Step 3: Identify the least significant feature
    least_significant_feature = p_values.idxmax()
    print(least_significant_feature)
    # Step 4: Remove the least significant feature
    X_s = X_s.drop(least_significant_feature, axis=1)

    # Step 5: Re-fit the model
    model = sm.OLS(y_s.astype(float), X_s.astype(float)).fit()

    # Step 6: Repeat until stopping criterion
    if p_values.max() < 0.05:  # Example stopping criterion (adjust as needed)
        break

# Print the final selected features
print("Selected features:", X_s.columns)
print(len(X_s.columns))

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:25:50.721042Z","iopub.execute_input":"2023-11-26T08:25:50.721380Z","iopub.status.idle":"2023-11-26T08:25:50.726205Z","shell.execute_reply.started":"2023-11-26T08:25:50.721356Z","shell.execute_reply":"2023-11-26T08:25:50.725254Z"}}
def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average = 'micro')

# %% [markdown]
# Decision Tree 

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:26:08.584335Z","iopub.execute_input":"2023-11-26T08:26:08.584664Z","iopub.status.idle":"2023-11-26T08:26:08.891956Z","shell.execute_reply.started":"2023-11-26T08:26:08.584638Z","shell.execute_reply":"2023-11-26T08:26:08.890437Z"}}
import matplotlib.pyplot as plt
train_errors = []
X_train, X_test, y_train, y_test = train_test_split(X_s,y_s,test_size=0.2,shuffle=True)

test_errors = []
depths = [i for i in range(2,20,2)]
for depth in depths:
    # Train a decision tree classifier with varying max depth
    clf = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    clf.fit(X_train, y_train)
    
    # Calculate training error
    y_pred_train = clf.predict(X_train)
    train_error =  calculate_f1(y_train, y_pred_train)
    train_errors.append(train_error)

    #calculate test error
    y_pred_test = clf.predict(X_test)
    test_error = calculate_f1(y_test,y_pred_test)
    test_errors.append(test_error)
    

# Plot the training error
plt.plot(depths, train_errors, marker='o')
plt.title('F1 score vs. Max Depth of Decision Tree')

plt.plot(depths, test_errors, marker='o')
plt.title('F1 score vs. Max Depth of Decision Tree')

plt.xlabel('Max Depth')
plt.ylabel('F1 score')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:26:17.215306Z","iopub.execute_input":"2023-11-26T08:26:17.215653Z","iopub.status.idle":"2023-11-26T08:26:17.709897Z","shell.execute_reply.started":"2023-11-26T08:26:17.215626Z","shell.execute_reply":"2023-11-26T08:26:17.709233Z"}}
import matplotlib.pyplot as plt
train_errors = []
test_errors = []
depths = [i for i in range(2,20,2)]
for depth in depths:
    # Train a decision tree classifier with varying max depth
    clf = RandomForestClassifier(max_depth=depth, criterion='entropy',n_estimators=10)
    clf.fit(X_train, y_train)
    
    # Calculate training error
    y_pred_train = clf.predict(X_train)
    train_error = calculate_f1(y_train, y_pred_train)
    train_errors.append(train_error)

    #calculate test er/ror
    y_pred_test = clf.predict(X_test)
    test_error = calculate_f1(y_test,y_pred_test)
    test_errors.append(test_error)
    

# Plot the training error
plt.plot(depths, train_errors, marker='o')
plt.title('F1 score vs. Max Depth of Random Forest Classifer')

plt.plot(depths, test_errors, marker='o')

plt.xlabel('Max Depth')
plt.ylabel(' F1 score')
plt.show()

# %% [code]


# %% [markdown]
# # Model Building
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:34:12.551119Z","iopub.execute_input":"2023-11-26T08:34:12.551555Z","iopub.status.idle":"2023-11-26T08:34:45.140260Z","shell.execute_reply.started":"2023-11-26T08:34:12.551508Z","shell.execute_reply":"2023-11-26T08:34:45.139070Z"}}
xgb_cv_scores = list()
lgbm_cv_scores = list()
cat_cv_scores = list()
decsion_cv_score = list()
rf_cv_score = list()
lgb_score = 0
kf=KFold(n_splits=5,shuffle=True,random_state=42)


for idx,(train_idx,test_idx) in enumerate(kf.split(X_s,y_s)):
    X_train,X_test=X.iloc[train_idx],X.iloc[test_idx]
    y_train,y_test=y.iloc[train_idx],y.iloc[test_idx]
    
    
    print('---------------------------------------------------------------')
    
    #Decison Tree
    clf = DecisionTreeClassifier(max_depth=8, criterion='entropy')
    clf.fit(X_train, y_train)
    dec_pred = clf.predict(X_test)
    dec_f1 = f1_score(y_test, dec_pred, average = 'micro')
    decsion_cv_score.append(dec_f1)
    print('Fold', idx+1, '==> Decision Tree  F1 score is ==>', dec_f1)

    #Random Forest Tree
    rf = RandomForestClassifier(max_depth=8, criterion='entropy',n_estimators=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred, average = 'micro')
    rf_cv_score.append(rf_f1)
    print('Fold', idx+1, '==> Random Forest Classifier  F1 score is ==>', rf_f1)
    
    
    #XGBClassifier
    params_xgb = {'objective': 'multi:softmax', 'num_class': 3, 'max_depth': 10, 'learning_rate': 0.01, 'colsample_bytree':.8}
    xgb_md = xgb.XGBClassifier(**params_xgb)
    xgb_md.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10000)
    xgb_pred=xgb_md.predict(X_test)
    xgb_f1 = f1_score(y_test, xgb_pred, average = 'micro')
    print('Fold', idx+1, '==> XGBoost  F1 score is ==>', xgb_f1)
    xgb_cv_scores.append(xgb_f1)
    
    
    #LightGBM
    param_grid = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.20,
            'colsample_bytree': 0.56,
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'random_state': 42,
            }

    lgbm_md = lgb.LGBMClassifier(**param_grid)
    lgbm_md.fit(X_train, y_train)
    lgbm_pred = lgbm_md.predict(X_test)   
    lgbm_f1 = f1_score(y_test, lgbm_pred, average = 'micro') 
    print('Fold', idx+1, '==> LightGBM  F1 score is ==>', lgbm_f1)
    lgbm_cv_scores.append(lgbm_f1)
        
    if lgbm_f1>lgb_score:
        lgb_score=lgbm_f1
        lgmb_best = lgbm_md
    #CatBoost
    cat_md = catboost.CatBoostClassifier(loss_function = 'MultiClass',
                                iterations = 500,
                                learning_rate = 0.01,
                                depth = 7,
                                random_strength = 0.5,
                                l2_leaf_reg = 5,
                                verbose = False, 
                                task_type = 'CPU')
    cat_md.fit(X_train, y_train)
    cat_pred = cat_md.predict(X_test)   
    cat_f1 = f1_score(y_test, cat_pred, average = 'micro')
    print('Fold', idx+1, '==> CatBoost  F1 score is ==>', cat_f1)
    cat_cv_scores.append(cat_f1)
    
    
print('---------------------------------------------------------------')
print('Average F1 Score of XGBoost model is:', np.mean(xgb_cv_scores))
print('Average F1 Score of LGBM model is:', np.mean(lgbm_cv_scores))
print('Average F1 Score of Catboost model is:', np.mean(cat_cv_scores))
print('Average F1 Score of Decision tree model is:', np.mean(decsion_cv_score))
print('Average F1 Score of Random Forest model is:', np.mean(rf_cv_score))



# %% [markdown]
# # Feature Importance

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:31:02.797613Z","iopub.execute_input":"2023-11-26T08:31:02.797936Z","iopub.status.idle":"2023-11-26T08:31:02.803771Z","shell.execute_reply.started":"2023-11-26T08:31:02.797912Z","shell.execute_reply":"2023-11-26T08:31:02.802855Z"}}
def feature_importances(model):
    feature_importance=pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
    return feature_importance
def plot_graph(feature_importance,model_name):
    plt.figure(figsize=(8,5))
    a=sns.barplot(x=feature_importance,y=feature_importance.index,palette='viridis')
    plt.xticks([])
    for j in ['right', 'top', 'bottom']:
        a.spines[j].set_visible(False)
    plt.title(f"{model_name} feature importances\n")
    plt.savefig(f"{model_name}.png")
    plt.show()
    plt.tight_layout()


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:31:03.172254Z","iopub.execute_input":"2023-11-26T08:31:03.172594Z","iopub.status.idle":"2023-11-26T08:31:04.544654Z","shell.execute_reply.started":"2023-11-26T08:31:03.172567Z","shell.execute_reply":"2023-11-26T08:31:04.543540Z"}}
plot_graph(feature_importances(xgb_md),'xgb')
plot_graph(feature_importances(lgbm_md),'lgbm_best')
plot_graph(feature_importances(cat_md),'cat')

# %% [code] {"execution":{"iopub.status.busy":"2023-11-25T15:06:43.079017Z","iopub.execute_input":"2023-11-25T15:06:43.080767Z","iopub.status.idle":"2023-11-25T15:06:43.514709Z","shell.execute_reply.started":"2023-11-25T15:06:43.080714Z","shell.execute_reply":"2023-11-25T15:06:43.513277Z"}}
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate the confusion matrix for the test set predictions
confusion_matric = confusion_matrix(y_test, lgbm_pred)

# ConfusionMatrixDisplay object with the confusion matrix and display labels
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matric, display_labels=["Dead", "Euthanized", "Lived"])

# Plot the confusion matrix
cm_display.plot()
plt.savefig("cfm lgbm")


# %% [code] {"execution":{"iopub.status.busy":"2023-11-25T15:08:16.060220Z","iopub.execute_input":"2023-11-25T15:08:16.060789Z","iopub.status.idle":"2023-11-25T15:08:16.423102Z","shell.execute_reply.started":"2023-11-25T15:08:16.060740Z","shell.execute_reply":"2023-11-25T15:08:16.421670Z"}}
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate the confusion matrix for the test set predictions
confusion_matric = confusion_matrix(y_test, xgb_pred)

# ConfusionMatrixDisplay object with the confusion matrix and display labels
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matric, display_labels=["Dead", "Euthanized", "Lived"])

# Plot the confusion matrix
cm_display.plot()
plt.show()
plt.savefig("cfm xgb")


# %% [code] {"execution":{"iopub.status.busy":"2023-11-25T15:07:02.675302Z","iopub.execute_input":"2023-11-25T15:07:02.675848Z","iopub.status.idle":"2023-11-25T15:07:03.112252Z","shell.execute_reply.started":"2023-11-25T15:07:02.675803Z","shell.execute_reply":"2023-11-25T15:07:03.110956Z"}}
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate the confusion matrix for the test set predictions
confusion_matric = confusion_matrix(y_test, cat_pred)

# ConfusionMatrixDisplay object with the confusion matrix and display labels
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matric, display_labels=["Dead", "Euthanized", "Lived"])

# Plot the confusion matrix
cm_display.plot()
plt.savefig("cfm cat")

plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2023-11-24T08:39:23.339189Z","iopub.execute_input":"2023-11-24T08:39:23.339687Z","iopub.status.idle":"2023-11-24T08:39:23.362788Z","shell.execute_reply.started":"2023-11-24T08:39:23.339650Z","shell.execute_reply":"2023-11-24T08:39:23.361058Z"}}
from sklearn.metrics import classification_report

# classification report comparing the true labels (y_test) with predicted labels (y_pred_test)
report = classification_report(y_test, y_predlgb)
print(report)


# %% [markdown]
# # Evaluation

# %% [code] {"execution":{"iopub.status.busy":"2023-11-24T06:40:38.476867Z","iopub.execute_input":"2023-11-24T06:40:38.477343Z","iopub.status.idle":"2023-11-24T06:40:38.482800Z","shell.execute_reply.started":"2023-11-24T06:40:38.477303Z","shell.execute_reply":"2023-11-24T06:40:38.481789Z"}}
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %% [markdown]
# Confusion Matrix
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-11-24T06:40:39.125953Z","iopub.execute_input":"2023-11-24T06:40:39.126406Z","iopub.status.idle":"2023-11-24T06:40:39.381991Z","shell.execute_reply.started":"2023-11-24T06:40:39.126370Z","shell.execute_reply":"2023-11-24T06:40:39.380656Z"}}
confusion_matric = confusion_matrix(y_test,y_predcat)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matric, display_labels = ["Dead","Euthanized","Lived"])

cm_display.plot()
plt.show()



# %% [code]
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predcat))


# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:35:37.397535Z","iopub.execute_input":"2023-11-26T08:35:37.397855Z","iopub.status.idle":"2023-11-26T08:35:37.403336Z","shell.execute_reply.started":"2023-11-26T08:35:37.397832Z","shell.execute_reply":"2023-11-26T08:35:37.402265Z"}}
test.drop(['lesion_2'],axis=1,inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:35:39.854020Z","iopub.execute_input":"2023-11-26T08:35:39.854367Z","iopub.status.idle":"2023-11-26T08:35:39.863438Z","shell.execute_reply.started":"2023-11-26T08:35:39.854341Z","shell.execute_reply":"2023-11-26T08:35:39.862463Z"}}
scaler.fit(test)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:35:40.729530Z","iopub.execute_input":"2023-11-26T08:35:40.729843Z","iopub.status.idle":"2023-11-26T08:35:40.740217Z","shell.execute_reply.started":"2023-11-26T08:35:40.729820Z","shell.execute_reply":"2023-11-26T08:35:40.739144Z"}}
scaler.transform(test)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:35:41.833972Z","iopub.execute_input":"2023-11-26T08:35:41.834358Z","iopub.status.idle":"2023-11-26T08:35:41.860351Z","shell.execute_reply.started":"2023-11-26T08:35:41.834330Z","shell.execute_reply":"2023-11-26T08:35:41.858848Z"}}
test_pred = catboost_model.predict(test)

# %% [markdown]
# # Submission
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-11-24T08:43:04.022809Z","iopub.execute_input":"2023-11-24T08:43:04.023278Z","iopub.status.idle":"2023-11-24T08:43:08.804591Z","shell.execute_reply.started":"2023-11-24T08:43:04.023244Z","shell.execute_reply":"2023-11-24T08:43:08.802759Z"}}
# lgbm_best.fit(X,y)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:35:45.073557Z","iopub.execute_input":"2023-11-26T08:35:45.073897Z","iopub.status.idle":"2023-11-26T08:35:45.097331Z","shell.execute_reply.started":"2023-11-26T08:35:45.073871Z","shell.execute_reply":"2023-11-26T08:35:45.096144Z"}}
T_pred = lgmb_best.predict(test)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:35:48.089462Z","iopub.execute_input":"2023-11-26T08:35:48.089824Z","iopub.status.idle":"2023-11-26T08:35:48.098646Z","shell.execute_reply.started":"2023-11-26T08:35:48.089798Z","shell.execute_reply":"2023-11-26T08:35:48.098024Z"}}
submission = pd.read_csv("/kaggle/input/playground-series-s3e22/sample_submission.csv")
T_preds = [1]*len(T_pred)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:35:48.969505Z","iopub.execute_input":"2023-11-26T08:35:48.970439Z","iopub.status.idle":"2023-11-26T08:35:48.975284Z","shell.execute_reply.started":"2023-11-26T08:35:48.970411Z","shell.execute_reply":"2023-11-26T08:35:48.974201Z"}}
for i in range(len(T_pred)):
    if T_pred[i]==1:
        T_preds[i]='euthanized'
    if T_pred[i]==0:
        T_preds[i]='died'
    if T_pred[i]==2:
        T_preds[i]='lived'

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:35:49.809505Z","iopub.execute_input":"2023-11-26T08:35:49.811913Z","iopub.status.idle":"2023-11-26T08:35:49.815488Z","shell.execute_reply.started":"2023-11-26T08:35:49.811883Z","shell.execute_reply":"2023-11-26T08:35:49.814878Z"}}
submission["outcome"]=T_preds

# %% [code] {"execution":{"iopub.status.busy":"2023-11-26T08:35:53.213560Z","iopub.execute_input":"2023-11-26T08:35:53.213901Z","iopub.status.idle":"2023-11-26T08:35:53.222213Z","shell.execute_reply.started":"2023-11-26T08:35:53.213875Z","shell.execute_reply":"2023-11-26T08:35:53.221000Z"}}
submission.to_csv("/kaggle/working/submission_lgbbest.csv",index=False)

# %% [code]
