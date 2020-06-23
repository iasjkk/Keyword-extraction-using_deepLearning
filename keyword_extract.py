import os
from nltk.tokenize import sent_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from seqeval.metrics import f1_score


train_path = "maui-semeval2010-train"
MAX_LEN = 75
bs = 32
tag2idx = {'B': 0, 'I': 1, 'O': 2}
tags_vals = ['B', 'I', 'O']
epochs = 4
max_grad_norm = 1.0
MODEL_DIR = f'keyword_extract_model.pt'



txt = sorted([f for f in os.listdir(train_path) if not f.endswith("-justTitle.txt") 
    and not f.endswith(".key") and not f.endswith("-CrowdCountskey")])
key = sorted([f for f in os.listdir(train_path) if f.endswith(".key")])


filekey = dict()
for i, k in enumerate(txt):
    filekey[key[i]] = k


def file_key_to_sents_and_labels(key):
    sentences = ""
    for line in open(train_path + "/" + filekey[key], 'r'):
        sentences += (" " + line.rstrip())
    tokens = sent_tokenize(sentences)
    key_file = open(train_path + "/" + str(key),'r')
    keys = [line.strip() for line in key_file]
    key_sent = []
    labels = []
    for token in tokens:
        z = ['O'] * len(token.split())
        for k in keys:
            if k in token:
                
                if len(k.split())==1:
                    try:
                        z[token.lower().split().index(k.lower().split()[0])] = 'B'
                    except ValueError:
                        continue
                elif len(k.split())>1:
                    try:
                        if (token.lower().split().index(k.lower().split()[0]) 
                            and token.lower().split().index(k.lower().split()[-1])):
                            z[token.lower().split().index(k.lower().split()[0])] = 'B'
                            for j in range(1, len(k.split())):
                                z[token.lower().split().index(k.lower().split()[j])] = 'I'
                    except ValueError:
                        continue
        for m, n in enumerate(z):
            if z[m] == 'I' and z[m-1] == 'O':
                z[m] = 'O'

        if set(z) != {'O'}:
            labels.append(z) 
            key_sent.append(token)
    return key_sent, labels

sents_labels_extraxt(filekey):
    sentences_ = []
    labels_ = []
    for key, value in filekey.items():
        s, l = file_key_to_sents_and_labels(key)
        sentences_.append(s)
        labels_.append(l)
    sentences = [item for sublist in sentences_ for item in sublist]
    labels = [item for sublist in labels_ for item in sublist]
    return sentences, labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


sentences, labels = sents_labels_extraxt(filekey)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]


input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")


attention_masks = [[float(i>0) for i in ii] for ii in input_ids]


tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)


tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)


train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

# model = model.cuda()


FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)




for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))


    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
torch.save(model.state_dict(), MODEL_DIR)


if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
    
MODEL_DIR = f'keyword_extract_model.pt'
model.load_state_dict(torch.load(MODEL_DIR, map_location=map_location))



model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


def keywordextract(sentence):
    text = sentence
    tkns = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english')) 
    tkns = [w for w in tkns if not w in stop_words]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tkns)
    segments_ids = [0] * len(tkns)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    model.eval()
    prediction = []
    logit = model(tokens_tensor, token_type_ids=None,
                                  attention_mask=segments_tensors)
    logit = logit.detach().cpu().numpy()
    prediction.extend([list(p) for p in np.argmax(logit, axis=2)])
    for k, j in enumerate(prediction[0]):
        if j==1 or j==0:
            print(tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k], j)


if __name__ == '__main__':
    text = '''Business groups and immigrant rights advocates are crying foul. 
    They say these restrictions will ultimately harm the economy, 
    and they're accusing the Trump administration of using the public health 
    crisis as a pretext to enact unnecessary immigration restrictions. 
    Unless you're an attorney or an immigrant with experience navigating 
    the US system, the alphabet soup of visas listed in Monday's 
    proclamation might be tough to decipher. The bottom line: 
    a wide range of workers, from au pairs to software engineers, 
    will be blocked from coming to the US at least until January. 
    And those restrictions could be extended. There are some exceptions. 
    Among them, the proclamation says officials will come up with 
    standards to let in people treating Covid-19 patients or 
    conducting research to help the US combat the pandemic. 
    It also will draft similar standards to admit people who are critical 
    to national security, are necessary to help the country's economic 
    recovery or are essential to the US food supply chain. And the new 
    measures don't apply to people who've already been issued valid visas. 
    But even so, the Migration Policy Institute estimates some 167,000 
    temporary workers will be kept out of the United States as a result 
    of these new restrictions, which take effect on Wednesday. Here's a 
    look at what kind of jobs are included in Trump's proclamation, and 
    how many people could be affected in each visa category.'''


    list_text = nltk.tokenize.sent_tokenize(text)
    for i in list_text:
      keywordextract(i)






