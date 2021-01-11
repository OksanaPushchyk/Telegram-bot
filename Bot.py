#!/usr/bin/env python
# coding: utf-8


import telebot
from telebot import types

from os.path import join as pjoin
import dill
from models.classifier import FakeNewsClassifier
import pandas as pd 
import re 
import string
from gensim.utils import tokenize as gensim_tokenize
import nltk  
from scipy.sparse import hstack
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score

from sklearn.model_selection import train_test_split
seed = 42

model_dir = 'models'
classifier_path = pjoin(model_dir, 'fake_news_classifier.pkl')

fn = FakeNewsClassifier(
    title_col = 'title',
    text_col = 'text')
fn.load(classifier_path)

TOKEN = 'my_token'

bot = telebot.TeleBot(TOKEN)

Subject = ''
Body = ''
ID = ''
Source_type = ''


@bot.message_handler(commands=['start'])
def start_chat(message):
    bot.send_message(message.chat.id, 'Hello! You can check articles here!')
    
@bot.message_handler(commands=['menu'])
def menu_command(message):
    keyboard = types.InlineKeyboardMarkup()
    key_analysis = types.InlineKeyboardButton(text='START ANALYSIS', callback_data='analysis')
    keyboard.add(key_analysis)
    bot.send_message(message.from_user.id, text='Choose what I need to do.', reply_markup=keyboard)
    
def enter_source(query):
    bot.answer_callback_query(query.id)
    keyboard = telebot.types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton(text='From plain text.', callback_data='from_plain_text'))
    keyboard.add(types.InlineKeyboardButton(text='From forward.', callback_data='from_forward'))
    bot.send_message(query.message.chat.id, text='Choose the source of data.', reply_markup=keyboard)
    
def set_subject(subject, id_subject):
    global Subject
    Subject = subject
    #bot.send_message(id_subject, Subject)
    #bot.register_next_step_handler(sent, menu_command)
    
def set_body(body, id_body):
    global Body
    Body = body
    global ID
    ID = id_body
    #bot.send_message(ID, Body)
    #bot.register_next_step_handler(sent, menu_command)
    
def set_source_type(source, id_source):
    global Source_type
    Source_type = source

#     bot.send_message(id_source, Source_type)
    #bot.register_next_step_handler(sent, menu_command)
    
    
def check_data(data, n):
    data = data.replace(' ', '')
    if len(data) > n:
        d = data[:n]
        flag = True
    else:
        d = data
        flag = False
    return d, flag

'''def init_model():
    fn = FakeNewsClassifier(
        title_col = 'title',
        text_col = 'text')
    
    fn.load(classifier_path)
    return fn'''

'''def classification_stats(y_true, y_pred, strategy='macro'):
    print('Accuracy : {:.5f}'.format(accuracy_score(y_true, y_pred)))
    print('Precision: {:.5f}'.format(precision_score(y_true, y_pred, average=strategy)))
    print('Recall   : {:.5f}'.format(recall_score(y_true, y_pred, average=strategy)))
    print('F1-Score : {:.5f}'.format(f1_score(y_true, y_pred, average=strategy)))'''
    
def get_value(sub, body, source, iid):
    # Here you have title and body of article.
    # Use model of prediction FAKE here.
    # Put probabulity of fake in prob_of_fake.
    pred_df = pd.DataFrame({'title': [sub], 'text':[body]})
    prob_of_fake = 1 - fn.predict(pred_df)[0]
    
    if (prob_of_fake >= 0) and (prob_of_fake < 0.26):
        massage = 'The probability of truth is ' + str(prob_of_fake) + '\nSeems almost definitely Fake.'
    elif (prob_of_fake >= 0.26) and (prob_of_fake <= 0.5):
        massage = 'The probability of truth is ' + str(prob_of_fake) + '\nSeems supposedly Fake.'
    elif (prob_of_fake > 0.5) and (prob_of_fake < 0.76):
        massage = 'The probability of truth is ' + str(prob_of_fake) + '\nSeems supposedly Genuine.'
    elif (prob_of_fake >= 0.76) and (prob_of_fake <= 1):
        massage = 'The probability of truth is ' + str(prob_of_fake) + '\nSeems almost definitely Genuine.'
    elif prob_of_fake == None:
        massage = 'Can’t be defined, please try smth else.'
        
    meta_information = get_metainfo(source)
    massage += meta_information
    bot.send_message(iid, massage)
    

    

def get_metainfo(source_type):
    template = '''
\n\n
Additional meta information:
input source : {source_type}'''
    
    return template.format(source_type=source_type)


@bot.callback_query_handler(func=lambda call: True)  
def iq_callback(query):  
    data = query.data  
    if data == 'analysis':  
        enter_source(query)
    elif data == 'from_plain_text':
        
        set_source_type('plain text input', query.message.chat.id)
        
        bot.answer_callback_query(query.id)
        sent = bot.send_message(query.message.chat.id, 'Enter subject of the data.')
        bot.register_next_step_handler(sent, enter_subject)
    elif data == 'from_forward':
        set_source_type('forwarded message input', query.message.chat.id)
        
        bot.answer_callback_query(query.id)
        sent = bot.send_message(query.message.chat.id, 'Forward message.')
        bot.register_next_step_handler(sent, enter_forward)
        
@bot.message_handler(content_types=['text'])
def enter_forward(message):
    if (message.text != '/start') and (message.text != '/menu') and (message.text != '/help'):
        if message.text != '':
            a = []
            a.append(message.text[:message.text.find('.') + 1])
            a.append(message.text[:message.text.find('?') + 1])
            a.append(message.text[:message.text.find('!') + 1])
            #length = min(len(a[0]), len(a[1]), len(a[2]))
            length = []
            for i in a:
                if i != '':
                    length.append(len(i))
            title = [i for i in a if len(i) == min(length)]
            body = message.text.replace(title[0], "")
            final_body = body.strip()
            bot.send_message(message.chat.id, 'Your subject:\n' + title[0])
            bot.send_message(message.chat.id, 'Your body:\n' + final_body)
            set_subject(title[0], message.chat.id)
            set_body(final_body, message.chat.id)
            get_value(Subject, Body, Source_type, message.chat.id)
        else:
            bot.send_message(message.chat.id, 'Article can not be empty.\nTry again.')
            bot.register_next_step_handler(sent, enter_forward)
    elif message.text == '/start':
        start_chat(message)
    elif message.text == '/menu':
        menu_command(message)

@bot.message_handler(content_types=['text'])
def enter_subject(message):
    if (message.text != '/start') and (message.text != '/menu') and (message.text != '/help'):
        if message.text != '':
            bot.send_message(message.chat.id, 'Your subject is ' + message.text)
            subject, flag = check_data(message.text, 1000)
            if flag:
                bot.send_message(message.chat.id, 'Subject has more than 1000 symbols.\nI deleted symbols after 1000’s.')
            sent = bot.send_message(message.chat.id, 'Enter body of the news.')
            set_subject(subject, message.chat.id)
            bot.register_next_step_handler(sent, enter_body)
        else:
            bot.send_message(message.chat.id, 'Subject can not be empty.\nTry again.')
            bot.register_next_step_handler(sent, enter_subject)
    elif message.text == '/start':
        start_chat(message)
    elif message.text == '/menu':
        menu_command(message)
    
@bot.message_handler(content_types=['text'])
def enter_body(message):
    if (message.text != '/start') and (message.text != '/menu') and (message.text != '/help'):
        if message.text != '':
            bot.send_message(message.chat.id, 'Your body:\n' + message.text)
            body, flag = check_data(message.text, 5000)
            if flag:
                bot.send_message(message.chat.id, 'Body has more than 5000 symbols.\nI deleted symbols after 5000’s.')
            set_body(body, message.chat.id)
            get_value(Subject, Body, Source_type, message.chat.id)
            
            #bot.register_next_step_handler(sent, menu_command)
        else:
            bot.send_message(message.chat.id, 'Body can not be empty.\nTry again.')
            bot.register_next_step_handler(sent, enter_subject)
    elif message.text == '/start':
        start_chat(message)
    elif message.text == '/menu':
        menu_command(message)
        

bot.polling(none_stop=True)



