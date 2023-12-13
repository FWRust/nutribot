import telebot
import pandas as pd
import numpy as np
import torch
from telebot import types
from peft import PeftConfig, PeftModel
from transformers import (
     AutoModelForCausalLM,
     AutoTokenizer,
     pipeline,
     GPTQConfig
 )
from googletrans import Translator

translator = Translator()


### Инициализация модели ######################################################################################
# Эта ячейка загружает модель в видеокарту, ни в коем случае ее нельзя запускать несколько раз за одну сессию,
# иначе в видюху загрузится несколько моделей и в ней начнет закачниватся видеопамять,
# если все таки нужно повторно запустить код, то сначала перезапустите среду чтобы освободить
# память от прошлой модели.
# По умолчанию модель занимает не более 10 ГБ VRAM, если занято больше значит вы что то делаете не так
#####################################################################################################################

base_model_name = "TheBloke/Llama-2-13B-chat-GPTQ"
adapter_model = '/home/ubuntu/model'

model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                              device_map={"": 0},
                                              quantization_config=GPTQConfig(bits=4,disable_exllama=True))
model = PeftModel.from_pretrained(model, adapter_model)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

### Параметры модели ###

pipe = pipeline('text-generation',model=model,tokenizer=tokenizer,
                 max_new_tokens=1024, # Максимальная длина ответа, сильное снижение может привести к обрезанию ответов
                 do_sample=True,   # не трогать
                 temperature=0.1,  # рандомность ответа, чем больше это число тем более неожиданные и креативные ответы будет давать модель, если число маленькое то ответы будут однообразными но уверенными.
                 top_p=0.95, #  тоже рандомность, чем меньше тем больше модель будет отходить от контекста.
                 top_k=40, # аналагично параметру выше, не рекомендуется менять.
                 repetition_penalty=1.15  # штраф за повторение, чем выше тем меньше модель будет повторять одни и те же слова/предложения, слишком высокое значение может привести к бреду/галлюцинациям
                )

# # ### Системный промпт, в нем вы обьясняете боту кто он по жизни и в чем его цель, что ему стоит делать а что нет, можно свободно изменять.

#system_prompt = "You are a helpful, respectful and honest assistant, you answer questions regarding food supplements and vitamins. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Answer the topic of the question, do not deviate from the topic of the question, do not philosophize, be precise and concise, use less than 150 words. Don't use any diagnoses or statistics in your answer. Use as few medical terms as possible. Do not ask questions back. Do not advise any vitamins."
system_prompt = "You are helpful assistant-nutritionist that answers patient's questions. Answer as short as possible."


# Берем результаты опросника и на основе них выводим возможные дефициты:
def show_results(message, hypothyroidism, insulinresistance, irondeficit, specific_id, db, user_id):
    if hypothyroidism == irondeficit == insulinresistance == 0:
        bot.send_message(message.chat.id, "Вы можете сдать следующие анализы(даны в табличке ниже), чтобы быть уверенными, что никаких проблем у Вас нет."
                                              "Результаты анализов впишите в данную табличку и отправьте боту")
        print_instructions(message)
        bot.send_document(message.chat.id, open('Анализы3.xlsx', 'rb'))
        bot.register_next_step_handler(message, get_analysis)
    else:
      db = db.drop(db.index[specific_id])
      db.to_csv("user_data.csv", index=False)
      with open('user_data.csv', 'a') as f:
        markup = types.ReplyKeyboardMarkup(resize_keyboard = True, one_time_keyboard = True)
        okButton = types.KeyboardButton("Понятно")
        markup.row(okButton)
        bot.send_message(message.chat.id, "У Вас возможны дефициты следующих элементов: ")
        if hypothyroidism > irondeficit or hypothyroidism > insulinresistance:
            bot.send_document(message.chat.id, open(r"Цинк.pdf", 'rb'))
            bot.send_document(message.chat.id, open(r"Йод.pdf", 'rb'))
            bot.send_document(message.chat.id, open(r"Селен.pdf", 'rb'))
            bot.send_document(message.chat.id, open(r"Жиры.pdf", 'rb'), reply_markup=markup)
            f.write(f"{user_id}, {1}, {MAX_QUESTIONS}, {hypothyroidism}, {insulinresistance}, {irondeficit}, {1}\n")
        elif irondeficit > insulinresistance or irondeficit >= hypothyroidism:
            bot.send_document(message.chat.id, open(r"Цинк.pdf", 'rb'))
            bot.send_document(message.chat.id, open(r"Хром.pdf", 'rb'))
            bot.send_document(message.chat.id, open(r"Магний.pdf", 'rb'), reply_markup=markup)
            f.write(f"{user_id}, {1}, {MAX_QUESTIONS}, {hypothyroidism}, {insulinresistance}, {irondeficit}, {2}\n")
        elif irondeficit == insulinresistance or insulinresistance > irondeficit or insulinresistance >= hypothyroidism:
            bot.send_document(message.chat.id, open(r"Кобаламин (B12).pdf", 'rb'))
            bot.send_document(message.chat.id, open(r"Фолиевая кислота.pdf", 'rb'))
            bot.send_document(message.chat.id, open(r"Медь.pdf", 'rb'))
            bot.send_document(message.chat.id, open(r"Витамин C.pdf", 'rb'))
            bot.send_document(message.chat.id, open(r"Железо.pdf", 'rb'), reply_markup = markup)
            f.write(f"{user_id}, {1}, {MAX_QUESTIONS}, {hypothyroidism}, {insulinresistance}, {irondeficit}, {3}\n")


# Получаем информацию о результатах опроса конкретного пользователя
def get_data(user_id, db, specific_id):
  hypothyroidism = list(db['Г'])[specific_id]
  insulinresistance = list(db['ИР'])[specific_id]
  irondeficit = list(db['ЖД'])[specific_id]
  return hypothyroidism,insulinresistance,irondeficit


# Получаем id пользователя в базе пользователей
def get_user_id_in_base(user_id):
  with open('user_data.csv') as f:
    db = pd.read_csv('user_data.csv')
    user_ids = list(db['user_id'])
    if user_id in user_ids:
      specific_id = user_ids.index(user_id)
    else:
      specific_id = None
  return specific_id, db

def print_instructions(message, markup=None):
  bot.send_message(message.chat.id, "Подготовка к анализам:\n"
                      "Общие правила подготовки и сдачи анализов крови на витамины и минералы:\n"
                      "●прийти в лабораторию заранее, чтобы привести в порядок эмоциональное и физическое состояние\n"
                      "●с момента последнего приема пищи должно пройти не менее восьми часов\n"
                      "●с утра разрешается пить воду, только чистую без добавок\n"
                      "●за неделю до сдачи (примерно) отказаться от употребления содержащих спирт напитков (в Т.Ч.\n"
                      "аптечной продукции)\n"
                      "●в день сдачи крови желательно воздерживаться от курения\n"
                      "●нельзя совмещать дни посещения лаборатории и физиотерапевтических (аппаратных) процедур исключить интенсивные физические нагрузки\n"
                      "Если Вы находитесь в состоянии сильных эмоциональных переживаний, возможно получение искаженных результатов.")


# Подгружаем все необходимые файлы
user_agreement = open(r'Пользовательское соглашение.pdf', 'rb')  # название

process_data = open(r'Обработка_персональных_данных.pdf', 'rb')

questions = open(r'questions.txt', 'rb')

zinc = open(r'Цинк.pdf', 'rb')

chrome = open(r'Хром.pdf', 'rb')

acid = open(r'Фолиевая кислота.pdf', 'rb')

test_results = open(r"Показания.xlsx", 'rb')

selen = open(r'Селен.pdf', 'rb')

copper = open(r'Медь.pdf', 'rb')

magnium = open(r'Магний.pdf', 'rb')

b12 = open(r'Кобаламин (B12).pdf', 'rb')

iodine = open(r'Йод.pdf', 'rb')

fats = open(r'Жиры.pdf', 'rb')

iron = open(r'Железо.pdf', 'rb')

vitamineC = open(r'Витамин C.pdf', 'rb')

#file_id = find_file()
db = open(r'user_data.csv', 'rb')

MAX_QUESTIONS = 33
token = '6330602631:AAGD-y1wboKQSXOkyJUWlV7UVXmNRVgPN90'
bot = telebot.TeleBot(token)
@bot.message_handler(content_types=['text', 'audio', 'document', 'animation', 'game', 'photo', 'sticker', 'video', 'video_note', 'voice', 'location', 'contact', 'venue', 'dice', 'new_chat_members', 'left_chat_member', 'new_chat_title',
                  'new_chat_photo', 'delete_chat_photo', 'group_chat_created', 'supergroup_chat_created', 'channel_chat_created', 'migrate_to_chat_id', 'migrate_from_chat_id', 'pinned_message', 'invoice', 'successful_payment',
                  'connected_website', 'poll', 'passport_data', 'proximity_alert_triggered', 'video_chat_scheduled', 'video_chat_started', 'video_chat_ended', 'video_chat_participants_invited', 'web_app_data',
                  'message_auto_delete_timer_changed', 'forum_topic_created', 'forum_topic_closed', 'forum_topic_reopened', 'forum_topic_edited', 'general_forum_topic_hidden', 'general_forum_topic_unhidden', 'write_access_allowed',
                 'user_shared', 'chat_shared', 'story'])
def sample(message):
#    global download_db, file_id
    user_id = message.from_user.id  # Получаем тг id пользователя
    questions = list(open('questions.txt', encoding='utf-8'))
    specific_id, db = get_user_id_in_base(message.from_user.id)
    # Блок с приветствием от бота
    if message.content_type == 'text' and (message.text.lower() == 'старт' or message.text.lower() == '/start'):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        but1 = types.KeyboardButton('✅ Начать')
        markup.add(but1)
        bot.send_message(message.chat.id,
                         "Здравствуйте! Я Онлайн нутрицолог, моя задача - помочь Вам"
                         " с решением возможных проблем со здоровьем путём применения"
                         " БАДов и витаминов. Приступим к работе?", parse_mode="html", reply_markup=markup)

    # Блок с пользовательским соглашением
    elif message.content_type == 'text' and message.text == '✅ Начать':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        but2 = types.KeyboardButton('Подтверждаю')
        markup.row(but2)
        bot.send_document(message.chat.id, open(r'Пользовательское соглашение.pdf', 'rb'))
        bot.send_document(message.chat.id, open(r'Обработка_персональных_данных.pdf', 'rb'))
        bot.send_message(message.chat.id, 'ℹ️ Прежде чем работать с ботом, обязательно'
                                            ' обратите внимание на пользовательское соглашение.'
                                            '\n \nНажимая кнопку "Подтверждаю", Вы автоматически считаетесь '
                                            'ознакомленными'
                                            ' и согласными с его условиями.', parse_mode='html', reply_markup=markup)


    # Блок со службой поддержки
    elif message.content_type == 'text' and message.text == 'Подтверждаю':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        button4 = types.KeyboardButton('✅ Хорошо')
        markup.row(button4)
        bot.send_message(message.chat.id, "ℹ️ Обратите внимание, что у нас есть Служба Поддержки,"
                                            " в которую Вы можете обратиться в случае возникновения каких-либо жалоб"
                                            " или вопросов."
                                            "\nЭлектронная почта Службы Поддержки : lunarfly_off@mail.ru"
                                            "\nСреднее время ответа Службы Поддержки занимает от 1 до 3 рабочих дней.",
                          parse_mode="html", reply_markup=markup)

    # !!!Пишите все, что должно быть после соглашения в блоке снизу
    elif message.content_type == 'text' and (message.text == '✅ Хорошо' or message.text == "Начать проходить заново"):
      specific_id, db = get_user_id_in_base(message.from_user.id)
      if specific_id != None:
        db = db.drop(db.index[specific_id])
        db.to_csv("user_data.csv", index=False)
      with open('user_data.csv', 'a') as f:
        f.write(f"{user_id}, {1}, {0}, {0}, {0}, {0}\n") # 1 1 0 0 0 -> 1 0 0 0 0
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
      button3 = types.KeyboardButton("Приступаем")
      markup.row(button3)
      bot.send_message(message.chat.id, "Итак, начнем с опросника. В нём будут называться различные симптомы, которые могут быть представлены у Вас."
                                        " Отвечайте да, если сталкиваетесь с их проявлением, либо нет, если симптом Вас не беспокоит."
                                        " Просим отвечать честно, только так мы сможем дать Вам рекомендации.",
                        reply_markup=markup)


    # Начало блока с опросником
    elif message.content_type == 'text' and message.text == 'Приступаем':
      specific_id, db = get_user_id_in_base(user_id)
      question_num = list(db['question'])[specific_id]
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
      buttonYes = types.KeyboardButton('Да')
      buttonNo = types.KeyboardButton('Нет')
      buttonStartAgain = types.KeyboardButton("Начать проходить заново")
      markup.row(buttonYes, buttonNo, buttonStartAgain)
      bot.send_message(message.chat.id, f"{questions[question_num]}",reply_markup=markup)

    # Открываем БД с данными о пользователе
    elif message.content_type == 'text' and (message.text == "Да" or message.text == "Нет"):
      specific_id, db = get_user_id_in_base(user_id)
      question_num = list(db['question'])[specific_id]
      hypothyroidism, insulinresistance, irondeficit = get_data(user_id, db, specific_id)

      if question_num < 33:
        if message.text == "Да":
          if question_num == 2:
            irondeficit += 1
          if question_num < 9:
            hypothyroidism += 1
          elif 9 <= question_num < 21:
            insulinresistance += 1
          elif 22 <= question_num < 34:
            irondeficit += 1
          question_num += 1
        else:
          question_num += 1
        # На последнем вопросе уходим сюда, и только тогда он его засчитает
        if question_num == 33:
          db = db.drop(db.index[specific_id])
          db.to_csv("user_data.csv", index=False)
          with open('user_data.csv', 'a') as f:
            f.write(f"{user_id}, {1}, {question_num}, {hypothyroidism}, {insulinresistance}, {irondeficit}\n")
            db = pd.read_csv('user_data.csv')
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
          agreeButton = types.KeyboardButton('Узнать результаты')
          markup.row(agreeButton)
          bot.send_message(message.chat.id, f"{questions[question_num]}",reply_markup=markup)
        else:
          markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
          buttonYes = types.KeyboardButton('Да')
          buttonNo = types.KeyboardButton('Нет')
          buttonStartAgain = types.KeyboardButton("Начать проходить заново")
          markup.row(buttonYes, buttonNo, buttonStartAgain)
          bot.send_message(message.chat.id, f"{questions[question_num]}",reply_markup=markup)

        # Обновляем БД после каждого вопроса
          db = db.drop(db.index[specific_id])
          db.to_csv("user_data.csv", index=False)
          with open('user_data.csv', 'a') as f:
            f.write(f"{user_id}, {1}, {question_num}, {hypothyroidism}, {insulinresistance}, {irondeficit}\n")

    # Блок, где мы выдаем результаты опросника
    elif message.content_type == 'text' and message.text == "Узнать результаты":
      specific_id, db = get_user_id_in_base(user_id)
      hypothyroidism, insulinresistance, irondeficit = get_data(user_id, db, specific_id)
      show_results(message, hypothyroidism, insulinresistance, irondeficit, specific_id, db, user_id)

    elif message.content_type == 'text' and message.text == "Понятно":
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard = True)
      continueButton1 = types.KeyboardButton("Продолжить")
      markup.row(continueButton1)
      bot.send_message(message.chat.id, "Если Вы хотите узнать, какой комплекс БАДов Вам необходим,\n"
                                        "нужно сдать анализы. Продолжаем?", reply_markup = markup)


    elif message.content_type == 'text' and message.text == 'Продолжить':
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard = True)
      continueButton2 = types.KeyboardButton("Прочитано")
      markup.row(continueButton2)
      print_instructions(message)
      bot.send_message(message.chat.id, "Пожалуйста, обращайте внимание на единицы измерений, которые указаны в таблице."
        "В случае, когда Вы записываете десятичное число через точку и таблица меняет его на дату, запишите это число"
        "через запятую. При записи чисел НЕ используйте пробелы(при указании диапазона используйте дефис).", reply_markup=markup)
    # Выдаем соответствующую табличку, куда записывать результаты
    elif (message.content_type == 'text' or message.content_type == 'document') and message.text == "Прочитано":
      specific_id, db = get_user_id_in_base(user_id)
      table_num = list(db['table_num'])[specific_id]
      bot.send_message(message.chat.id, "В данной ниже таблице вместо нулей заполните Ваши результаты анализов."
                       "Удостоверьтесь, что Ферритин, ТТГ и инсулин заполнены!")
      if table_num == 1:
        bot.send_document(message.chat.id, open('analysis3.xlsx', 'rb'))
      elif table_num == 2:
        bot.send_document(message.chat.id, open('analysis1.xlsx', 'rb'))
      else:
        bot.send_document(message.chat.id, open('analysis2.xlsx', 'rb'))
      bot.register_next_step_handler(message, get_analysis) #переделал из get_analysis в request_for_model потому что я тупой и ниче не понял и я заебался, потом исправлю

    else:
      bot.send_message(message.chat.id, "Пожалуйста, отвечайте корректно. Используйте кнопки, предложенные под полем для ответа.")
# Получаем анализы от пользователя
def get_analysis(message):
  try:
    file_name = message.document.file_name
    result_table = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(result_table.file_path)
    with open(file_name, 'wb') as new_file:
      new_file.write(downloaded_file)
    results_to_compare = pd.DataFrame(pd.read_excel(file_name))
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard = True)
    manButton = types.KeyboardButton("Мужчина")
    womanButton = types.KeyboardButton("Женщина")
    markup.row(manButton, womanButton)
    bot.send_message(message.chat.id, "Для корректных результатов, необходимо узнать Ваш пол.", reply_markup = markup)
    bot.register_next_step_handler(message, process_analysis, results_to_compare)

  except Exception:
    bot.reply_to(message, "Упс, произошла ошибка, попробуйте еще раз!")
    bot.register_next_step_handler(message, get_analysis)

def process_analysis(message, results_to_compare):
  specific_id, db = get_user_id_in_base(message.from_user.id)
  hypothyroidism, insulinresistance, irondeficit = get_data(message.from_user.id, db, specific_id)
  db = db.drop(db.index[specific_id])
  db.to_csv("user_data.csv", index=False)
  with open('user_data.csv', 'a') as f:
    f.write(f"{message.from_user.id}, {1}, {MAX_QUESTIONS}, {hypothyroidism}, {insulinresistance}, {irondeficit}, {0}, {message.text}\n")
    db = pd.read_csv('user_data.csv')
  result_base = pd.read_excel("Показания.xlsx")
  main_analysis = {}
  for i in range(results_to_compare.count()[0]):
    row = list(results_to_compare.loc[i])
    if row[1] != 0 and not(pd.isnull(row[1])):
      if row[0] == "Инсулин" or row[0] == "ТТГ" or row[0] == "Ферритин":
        if row[1].is_integer() or isinstance(row[1], np.floating):
          main_analysis[row[0]] = row[1]


  # Саша, здесь есть повторяющиеся строки. Если у тебя есть идея, как от них избавиться, то давай :))) /// в ф-ию var_sender вынес все эти операции (она выше в комменнтариях).
  if ("Инсулин" not in main_analysis) or ("Ферритин" not in main_analysis) or ("ТТГ" not in main_analysis):
    bot.send_message(message.chat.id, "Вы ввели некорректные данные в таблице, попробуйте еще раз!"
                     "Удостоверьтесь, что Ферритин, ТТГ и инсулин заполнены!")
    bot.register_next_step_handler(message, get_analysis)
  elif 50 <= main_analysis["Ферритин"] <= 150 and 2 <= main_analysis["Инсулин"] <= 6 and 0.4 <= float(main_analysis["ТТГ"]) <= 2.0:
    bot.send_document(message.chat.id, open(r"Вариант4.pdf", 'rb'))
    bot.send_message(message.chat.id, "Если у вас остались вопросы, можете задать их в свободной форме ниже, на них Вам ответит искусственный интеллект[beta].\n"
    "Начинайте спрашивать прямо сейчас. Если у Вас нет вопросов или Вы захотите остановиться, напишите СТОП.")
    bot.register_next_step_handler(message, ask_model)
  else:
    if main_analysis["Ферритин"] <= main_analysis["Инсулин"] or main_analysis["Инсулин"] >= main_analysis["ТТГ"]:
      bot.send_document(message.chat.id, open(r"Вариант1.pdf", 'rb'))
      bot.send_message(message.chat.id, "Если у вас остались вопросы, можете задать их в свободной форме ниже, на них Вам ответит искусственный интеллект[beta].\n"
    "Начинайте спрашивать прямо сейчас. Если у Вас нет вопросов или Вы захотите остановиться, напишите СТОП.")
      bot.register_next_step_handler(message, ask_model)
    elif  main_analysis["Ферритин"] >= main_analysis["ТТГ"] or main_analysis["Ферритин"] > main_analysis["Инсулин"]:
      bot.send_document(message.chat.id, open(r"Вариант2.pdf", 'rb'))
      bot.send_message(message.chat.id, "Если у вас остались вопросы, можете задать их в свободной форме ниже, на них Вам ответит искусственный интеллект[beta].\n"
    "Начинайте спрашивать прямо сейчас. Если у Вас нет вопросов или Вы захотите остановиться, напишите СТОП.")
      bot.register_next_step_handler(message, ask_model)
    else:
      bot.send_document(message.chat.id, open(r"Вариант3.pdf", 'rb'))
      bot.send_message(message.chat.id, "Если у вас остались вопросы, можете задать их в свободной форме ниже, на них Вам ответит искусственный интеллект[beta].\n"
    "Начинайте спрашивать прямо сейчас. Если у Вас нет вопросов или Вы захотите остановиться, напишите СТОП.")
      bot.register_next_step_handler(message, ask_model)

def ask_model(user_prompt: str):
  bot.send_message(message.chat.id, 'Бот думает, подождите...')
  message = user_prompt
  if user_prompt.text.lower() != 'стоп':
    user_prompt = translator.translate(user_prompt.text).text
    output = pipe(f'''[INST] <>
    {system_prompt}<>
    {user_prompt}[/INST]''', return_full_text = False)[0]['generated_text']
    translated_output = translator.translate(output, src="en", dest="ru")
    bot.send_message(message.chat.id, translated_output.text)
    bot.register_next_step_handler(message, ask_model)
  else:
    againButton = types.KeyboardButton("Начать проходить заново")
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard = True)
    markup.row(againButton)
    bot.send_message(message.chat.id, "Вы можете начать проходить бота заново.", reply_markup = markup)
    bot.register_next_step_handler(message, sample)


bot.infinity_polling()
