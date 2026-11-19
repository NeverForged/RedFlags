import re
import nltk
import time
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import discord
from dotenv import load_dotenv

#RED FLAGS BOT

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Define intents (necessary for modern bots)
intents = discord.Intents.default()
intents.message_content = True  # Enable the bot to read message content
client = discord.Client(intents=intents)

# FUNCTIONS
def save_model_bundle(filepath, model_score, gatekeeper, models, gatekeeper_thresh, 
                      thresholds, nounVectorizer, gramVectorizer, charVectorizer, scaler):
    """Saves the trained models, vectorizers, scaler, and thresholds to a single file."""
    bundle = {
        'model_score': model_score,
        'gatekeeper': gatekeeper,
        'models': models,
        'gatekeeper_thresh': gatekeeper_thresh,
        'thresholds': thresholds,
        'nounVectorizer': nounVectorizer,
        'gramVectorizer': gramVectorizer,
        'charVectorizer': charVectorizer,
        'scaler': scaler
    }
    joblib.dump(bundle, filepath)
    print(f"Model bundle successfully saved to {filepath}")

def load_model_bundle(filepath):
    """Loads the model bundle and returns the components in their original order."""
    bundle = joblib.load(filepath)
    print(f"Model bundle loaded from {filepath}")
    return (
        bundle['model_score'],
        bundle['gatekeeper'],
        bundle['models'],
        bundle['gatekeeper_thresh'],
        bundle['thresholds'],
        bundle['nounVectorizer'],
        bundle['gramVectorizer'],
        bundle['charVectorizer'],
        bundle['scaler']
    )
    
def preprocess_text(series):
     # @somone to @
    lst = []
    lstn = []
    lstw  = []
    length = len(series)
    for i, text in enumerate(series):
        # mark @, # and links
        text = ' '.join(['_at_someone_' if '@' in word else word for word in text.split()])
        text = ' '.join(['_link_' if 'http' in word or 'www.' in word else word for word in text.split()])
        text = ' '.join(['_hashtag_' if '#' in word else word for word in text.split()])
        # remove all other punctuation
        text = ''.join([word.lower() for word in text if word not in string.punctuation.replace('@','').replace('#','').replace('_','')])

        # the real work, pos tagging to reduce Nouns and other things that don't matter
        tokens = []
        nouns = []
        weird = []
        for a in nltk.pos_tag(text.split()):


            if bool(re.search(r'[^\w\s]', a[0])):
                weird.append(nltk.PorterStemmer().stem(a[0]))

            # keep these as-is for reasons
            if '_link_' in a[0] or '_at_someone_' in a[0] or '_hashtag_' in a[0]:
                tokens.append(a[0])
                nouns.append(a[0])
            # lemmatize these to reduce parts of speach issues
            elif 'NN' in a[1] or 'JJ' in a[1] or 'RB' in a[1] or 'V' in a[1]:
                tokens.append(a[1])
                try:
                    nouns.append(a[0])
                except:
                    print('Recursion error',a)
            else:
                tokens.append(a[1])
                #nouns.append(nltk.PorterStemmer().stem(a[0]))
        lst.append(' '.join(tokens))
        lstn.append(' '.join(nouns))
        lstw.append(' '.join(weird))
        if i%100 == 0:
            print("\r{:.2f}% Complete".format(100*((i+1)/length), a), end="", flush=True)
    print("\r100.00% Complete".format(100*((i+1)/length)), end="", flush=True)
    return lst, lstn, lstw

def vectorize_data(df, nounVectorizer, gramVectorizer, charVectorizer, scaler):
    gV = gramVectorizer.transform(df['pos'])
    nV = nounVectorizer.transform(df['stems'])
    nC = charVectorizer.transform(df['comment_text'])
    transformed_data = hstack([gV, nV, nC])

    if scaler != None:
        scaler.fit(transformed_data)
    return transformed_data

def fit_vectorizers(df):
    # gram is based on the work I did in howRussian
    gramVectorizer =  TfidfVectorizer(ngram_range=(2,5)
                                        , min_df = 10
                                        , max_df = .7
                                        , stop_words = None
                                        , sublinear_tf=True
                                        , lowercase=False              # Protect POS tag case sensitivity
                                        , token_pattern=r'\S+'          # Tokenize strictly by space-separated tags/tokens
                                       )
    gramVectorizer.fit(df['pos'])
    
    
    
    # a simple vectorizer on the nouns to include specific words because reasons, ideal at word and bigram
    nounVectorizer = TfidfVectorizer(ngram_range=(1, 2)                
                                        , tokenizer=tokenize              
                                        , strip_accents='unicode'            
                                        , stop_words='english'               
                                        , sublinear_tf=True  # Replaces TF count with 1 + log(TF)
                                        , min_df=3                           
                                        , max_df=0.9                           
                                    )
    nounVectorizer.fit(df['stems'])


    # char_vec to catch weird mispellings...
    charVectorizer = TfidfVectorizer(ngram_range=(2, 6)               
                                        , analyzer='char'                   
                                        , sublinear_tf=True                 
                                        , strip_accents='unicode'           
                                        , min_df=3                          
                                        , max_features=50000                  
                                    )
    charVectorizer.fit(df[df['flag']==1]['punc'])

    
    gV = gramVectorizer.transform(df['pos'])
    nV = nounVectorizer.transform(df['stems'])
    nC = charVectorizer.transform(df['punc'])
    transformed_data = hstack([gV, nV, nC])

    # scaler = StandardScaler(with_mean=False)
    scaler = None
    # scaler.fit(transformed_data)
    return gramVectorizer, nounVectorizer, charVectorizer, scaler

def set_threshold(model = None, models = None, col = None, df=None, Xt=None):
    testing_gate = pd.DataFrame()
    testing_gate['id'] = df['id']
    cols = ['tobscene', 'tthreat', 'tinsult', 'tidentity_hate']
    if col != None and col != 'flag':
        for i, model in enumerate(models):
             testing_gate[cols[i]] = model.predict_proba(Xt)[:, 1]
    else:
        testing_gate['tflag'] = model.predict_proba(Xt)[:, 1]
    
    ndf = df.copy().merge(testing_gate, on='id')

    cols = ['obscene', 'threat', 'insult', 'identity_hate']
    if col == None:
        col = 'flag'
    if model == None:
        model = models[cols.index(col)]

    if col == 'flag':
        # Get false positive rates, true positive rates, and thresholds
        fpr, tpr, thresholds = roc_curve(ndf[col], ndf['t'+col])
        j_scores = tpr - fpr
        best_idx = j_scores.argmax()
        best_threshold = thresholds[best_idx]
        return best_threshold
    else:
        precision, recall, thresholds = precision_recall_curve(ndf[col], ndf['t' + col])
        
        # Calculate F1 score for each threshold
        # (Adding 1e-10 prevents division by zero errors)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find the index of the highest F1 score
        best_idx = f1_scores.argmax()
        best_threshold = thresholds[best_idx]
        if best_threshold > 0.5:
            best_threshold = 0.5
        return best_threshold
    
def test_models(gatekeeper, models, gthresh, mthresh, ndf, Xt): 
    # need the percents
    ndf['tflag'] = gatekeeper.predict_proba(Xt)[:, 1]
    cols = ['tobscene', 'tthreat', 'tinsult', 'tidentity_hate']
    for i, model in enumerate(models):
         ndf[cols[i]] = model.predict_proba(Xt)[:, 1]
    cols = ['obscene', 'threat', 'insult', 'identity_hate']
   
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    
    for i, row in ndf.iterrows():
        if row['flag'] == 0 and row['tflag'] < gthresh:
            true_negatives +=1
        elif row['flag'] == 0 and row['tflag'] >= gthresh:
            false_positives += 1
        elif row['flag'] == 1 and row['tflag'] < gthresh:
            false_negatives += 1
        else:
            checkfn = 0
            checkfp = 0
            for col in cols:
                if row[col] == 0 and row['t'+col] >= mthresh[cols.index(col)]:
                    checkfp += 1
                elif row[col] == 1 and row['t'+col] <  mthresh[cols.index(col)]:
                    checkfn += 1
            if checkfn >= 1:
                false_negatives += 1
    
            if checkfp >= 1:
                false_positives += 1
            if checkfn == 0 and checkfp == 0:
                true_positives += 1
  
    print('TP: {} TN: {} FP: {} FN: {}'.format(true_positives, true_negatives, false_positives, false_negatives))

def train_model(Xtrain, Xt, train_df, test_df, col, verbose=True):
    
    y = train_df[col]
    yt = test_df[col]
    
    model = LogisticRegression(
        C = 0.5
        , penalty='l2'
        , solver='lbfgs'
        , class_weight='balanced'
        , n_jobs = -1
    )
    
    model.fit(Xtrain,y)

    # Calculate probabilities
    y_pred_probs = model.predict_proba(Xtrain)[:, 1]
    yt_pred_probs = model.predict_proba(Xt)[:, 1]

    thresh = 0.35
    
    # Evaluate score based on custom threshold rather than default 0.5
    y_preds = (y_pred_probs >= thresh).astype(int)
    yt_preds = (yt_pred_probs >= thresh).astype(int)
    custom_acc = (yt_preds == yt).mean()
    
    if verbose:
        print('Training Data {:.3f}, Test Data Score (Thresh={:.2f}): {:.3f}'.format(
               (y_preds == y).mean(), thresh, custom_acc
            ))
        # Calculate ROC-AUC for the single binary label
        auc_score = roc_auc_score(y, y_pred_probs)
        
        auc_scoret = roc_auc_score(yt, yt_pred_probs)
        print(f"TRAINING ROC-AUC Score: {auc_score:.4f}, Test ROC-AUC Score: {auc_scoret:.4f}")
    return model

def build_model(model_score = 0.0, 
                ogatekeeper = None, 
                omodels = None, 
                ogatekeeper_thresh = 0, 
                othresholds = [0,0,0,0,0,0], 
                onounVectorizer=None, 
                ogramVectorizer=None, 
                ocharVectorizer=None, 
                oscaler=None):

    print('oldscore', model_score)

    stopword = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()

    # get and process data
    df = pd.read_csv('data/processed_data.csv')

    #toxic used here to set flag...
    cols = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['flag'] = df[cols].max(axis=1)
    df['pos'], df['stems'], df['punc'] = preprocess_text(df['comment_text'])
    
    # split the data
    train_df, test_df = train_test_split(df, test_size=0.2)
    gramVectorizer, nounVectorizer, charVectorizer, scaler = fit_vectorizers(train_df)
    
    # flag_df
    temp1 = train_df[train_df['flag'] == 1]
    # get the last 25000 comemnts in each group, focus on newer neg comments
    temp_ = pd.DataFrame()
    for col in ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        temp_ = pd.concat([temp_, temp1[temp1[col]==1].tail(25000)])
    print('temp_', temp_.shape)
    temp1 = temp_.drop_duplicates()
    print('temp1', temp1.shape)
   
    # twice as many non-flagged comments...
    bal_num = test_df[test_df['flag']==0].shape[0]/test_df[test_df['flag']==1].shape[0]
    temp2 = train_df[train_df['flag'] == 0].sort_values('pred', ascending=False).head(int(temp1.shape[0]*bal_num))
    flag_df = pd.concat([temp1, temp2]).sort_index().reset_index()
    tox_df = temp1.reset_index()
    test_flag_df = test_df[test_df['flag']==1].reset_index()

    # vectorize!
    Xf = vectorize_data(flag_df, nounVectorizer, gramVectorizer, charVectorizer, scaler)
    Xt = vectorize_data(test_df, nounVectorizer, gramVectorizer, charVectorizer, scaler)
    Xx = vectorize_data(tox_df, nounVectorizer, gramVectorizer, charVectorizer, scaler)
    Xtf = vectorize_data(test_flag_df, nounVectorizer, gramVectorizer, charVectorizer, scaler)

    # train the models
    gatekeeper = train_model(Xf, Xt, flag_df, test_df, col='flag', verbose=True)
    models = []
    for col in ['obscene', 'threat', 'insult', 'identity_hate']:
        models.append(train_model(Xx, Xtf, tox_df, test_flag_df, col=col, verbose=True))
        
    gatekeeper_thresh = set_threshold(model = gatekeeper, models = None, col = 'flag', df=test_df, Xt=Xt)
    thresholds = []
    [thresholds.append(set_threshold(model = None, models = models, col = col, df=test_flag_df, Xt=Xtf)) for col in ['obscene', 'threat', 'insult', 'identity_hate']]
    test_models(gatekeeper, models, gatekeeper_thresh, thresholds, ndf=test_df, Xt=Xt)

    # check score...
    scores  = []
    yt = test_df['flag']
    gscore = score_model(gatekeeper,Xt,yt,gatekeeper_thresh)

    for i, col in enumerate(['obscene', 'threat', 'insult', 'identity_hate']):
        yt = test_flag_df[col]
        scores.append(score_model(models[i],Xtf, yt,thresholds[i]))
    print(gscore, scores)
    mscore = sum(scores)/len(scores)
    # gatekeeper is more important...
    score = (gscore+mscore)/2
    print('newscore', score)
    if score >= model_score:  # = so newer models are preferred...
        #redo preds for current model
        X = vectorize_data(df, nounVectorizer, gramVectorizer, charVectorizer, scaler)
        df['pred'] = gatekeeper.predict_proba(X)[:, 1]
        cols = ['id', 'comment_text', 'toxic',	'obscene',	'threat',	'insult',	'identity_hate', 'pred', 'date']
        df[cols].to_csv('data/processed_data.csv', index=False)
        return score, gatekeeper, models, gatekeeper_thresh, thresholds, nounVectorizer, gramVectorizer, charVectorizer, scaler
    else:
        return model_score, ogatekeeper, omodels, ogatekeeper_thresh, othresholds, onounVectorizer, ogramVectorizer, ocharVectorizer, oscaler

def score_model(model, Xt, yt, thresh):
    # Calculate probabilities
    yt_pred_probs = model.predict_proba(Xt)[:, 1]
    # Evaluate score based on custom threshold rather than default 0.5
    yt_preds = (yt_pred_probs >= thresh).astype(int)
    return (yt_preds == yt).mean()

def tokenize(s):
    re_tok = re.compile(f'([{string.punctuation}])')
    # Splits words and isolates punctuation marks as individual tokens
    return re_tok.sub(r' \1 ', s).split()

def evaluate_post(post, gatekeeper, models, gatekeeper_thresh, thresholds, nounVectorizer, gramVectorizer, charVectorizer, scaler): 
    df = pd.DataFrame()
    df['comment_text'] = [post]
    df['pos'], df['stems'], df['weird'] = preprocess_text(df['comment_text'])
    X = vectorize_data(df, nounVectorizer, gramVectorizer, charVectorizer, scaler)
    cols = ['Obscene', 'Threat', 'Insult', 'Identity-Based Hate Speach']
    check = gatekeeper.predict_proba(X)[:, 1][0]
    response = []
    if check >= gatekeeper_thresh:
        response.append(('Toxic',check))
        for i, model in enumerate(models):
            check_ = model.predict_proba(X)[:, 1][0]
            if check_ > thresholds[i]:
                response.append((cols[i], check_))
            else:
                response.append((cols[i], 0))
    else:
        response = [(col, 0) for col in cols]
    return check, response


(model_score, gatekeeper, models, gatekeeper_thresh, 
 thresholds, nounVectorizer, gramVectorizer, charVectorizer, scaler) = load_model_bundle('redflags_model.joblib')
 
@client.event
async def on_ready():
    """Triggered when the bot is connected to Discord."""
    print(f'{client.user} has connected to Discord!')
    

@client.event
async def on_message(message):
    """Triggered every time a message is sent in a channel."""

    if message.author == client.user:
        return # Prevents the bot from responding to itself
        
    check, response = evaluate_post(message.content, gatekeeper, models, gatekeeper_thresh, thresholds, nounVectorizer, gramVectorizer, charVectorizer, scaler)  
    lst = []
    if response[0][1] > gatekeeper_thresh:
        await message.add_reaction("🚩")
        for i, a in enumerate(response):
            if (i!=0 and a[1] >= thresholds[i-1]) or (i==0 and a[1] >= gatekeeper_thresh):
                lst.append('{}: {:.2f}%'.format(a[0], a[1]*100))
        if len(lst) >= 1:
            flag = 'Red Flags: ' + ', '.join(lst)
            try:
                thread = await message.create_thread(
                        name=f"Analysis: {message.content[:50]}",  # Thread title (max 100 chars)
                        auto_archive_duration=60  # Archive after 60 mins of inactivity (60, 1440, 4320, or 10080)
                    )
                await thread.send(flag)
            except:
                await message.reply(flag)
# Run the bot with your token
client.run(TOKEN)
